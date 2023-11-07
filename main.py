import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class QuantumEmbed(torch.nn.Module):
    '''
        Embed tokens as quantum measurements
        dim: physical dimension
        voc_size: size of vocabulary dictionary
    '''
    
    def __init__(self, dim=2, voc_size=6, n=2):
        super().__init__()
        A = torch.randn(voc_size, dim, dim, dtype=torch.cfloat, device=device)
        self.A = torch.nn.Parameter(A)
        self.d = dim
        self.v = voc_size
        self.reset(one_hot=False)
        
    def reset(self, one_hot):
        if one_hot:
            assert self.d == self.v, 'physical dimension must equal vocabulary size for one hot embedding'
            self._M = torch.zeros((self.v, self.d, self.d), device=device).fill_diagonal_(1).to(torch.cfloat)
        else:
            self._M = None
                
    @property
    def M(self):
        if self._M is None:
            A = torch.linalg.qr(self.A.view(-1, self.d))[0].view(self.A.shape) # A = QR Q^+ @ Q = 1
            self._M = A.conj().mT @ A 
        return self._M
        
    def forward(self, token):
        # token: list of int ranging from 0 to voc_size - 1
        if torch.is_tensor(token):
            token = token.long().tolist()
        if len(token) == 0:
            return None
        assert max(token) < self.v, f'max(token)={max(token)} is out of range.'
        return self.M[token]
    
class MPS(torch.nn.Module):
    '''
        Create a MPS state with trainable parameters
    
    '''
    
    def __init__(self, n, phy_dim, bond_dim, voc_size=6):
        super().__init__()
        self.emb = QuantumEmbed(phy_dim, voc_size, n)
        self.phy_dim = phy_dim
        self.bond_dim = bond_dim
        self.n = n # number of spins
        self.mps_blocks = self.set_left_param()
        self.reset()
        
    @property
    def full_tensor(self):
        '''
        Warning: exponential complexity -> phy_dim ^ n
        '''
        T = self.mps_blocks[0]
        for i in range(self.n-1):
            T = torch.tensordot(T, self.mps_blocks[i+1], dims=([-1], [1]))
        return T.squeeze(1).squeeze(-1) # ((phy_dim, ) * n)
        
    @property
    def tok_all(self):
        return self.emb([i for i in range(self.emb.v)])
        
    def set_left_param(self):
        mps = []
        in_dim = 1
        for i in range(self.n):
            if i < self.n-1:
                L = torch.randn(self.phy_dim, in_dim, self.bond_dim, dtype=torch.cfloat, device=device)
            else:
                L = torch.randn(self.phy_dim, in_dim, 1, dtype=torch.cfloat, device=device)
            Q, R = torch.linalg.qr(L.view(-1, L.shape[-1]))
            mps.append(torch.nn.Parameter(Q.view(self.phy_dim, -1, Q.shape[-1])))
            in_dim = Q.shape[-1]
        return mps
    
    def QR(self):
        for i in range(self.n):
            self.mps_blocks[i].data = torch.linalg.qr(self.mps_blocks[i].data.view(-1, self.mps_blocks[i].shape[-1]))[0].view(self.mps_blocks[i].shape)
        
    def reset(self, one_hot=False):
        self._zipped_blocks = None
        self.QR()
        self.emb.reset(one_hot)
    
    # evaluate probability of token, or marginal probability condition on token (incomplete)
    def prob(self, token):
        Ms = self.emb(token) # (n, d, d)
        assert len(token) <= self.n, f'len(token) cannot be larger than number of qbits.'
        out = torch.tensor([[[[1.]]]], dtype=torch.cfloat, device=device) #(1, 1, 1, 1)
        for i in range(len(token)):
            zipped = torch.einsum(self.mps_blocks[i].conj(), [0, 1, 2], Ms[i], [0, 3], self.mps_blocks[i], [3, 4, 5], [1, 2, 5, 4])
            out = torch.einsum(out, [0, 1, 2, 3], zipped, [1, 4, 5, 2], [0, 4, 5, 3])
            # (bond_dim(left), bond_dim(right), bond_dim(right), bond_dim(left))
        if len(token) == self.n:
            out = out.view(-1)#.real
        return out
      
    # list of zipped mps blocks
    @property
    def zipped_blocks(self):
        if self._zipped_blocks is None:
            self._zipped_blocks = []
            for block in self.mps_blocks:
                zipped_block = torch.einsum(block.conj(), [0, 1, 2], block, [0, 3, 4], [1, 2, 4, 3])
                # (bond_dim(left), bond_dim(right), bond_dim(right), bond_dim(left))
                self._zipped_blocks.append(zipped_block)
        return self._zipped_blocks
    
    
    '''
    Note: this part should be reconstructed by sampling from right to left for a left-isometric mps
    '''
    # perform one step sampling condintion on incomplete token
    # T -> already contracted part (bond_dim(left), bond_dim(right), bond_dim(right), bond_dim(left))
    # site -> new sample site
    def sample_step(self, T=None, site=0):
        if T is None:
            T = torch.tensor([[[[1.]]]], dtype=torch.cfloat, device=device) #(1, 1, 1, 1)
        # (phy_dim, phy_dim, bond_dim(left), bond_dim(right), bond_dim(right), bond_dim(left))
        T = torch.einsum(T, [0, 1, 2, 3], self.mps_blocks[site].conj(), [4, 1, 5], self.mps_blocks[site], [6, 2, 7], [4, 6, 0, 5, 7, 3])
        out_T = T.clone() # save the output T for the next step sampling
        for i in range(self.n-site-1):
            T = torch.einsum(T, [0, 1, 2, 3, 4, 5], self.zipped_blocks[i+site+1], [3, 6, 7, 4], [0, 1, 2, 6, 7, 5])
        T = T.view(self.phy_dim, self.phy_dim)
        probs = torch.tensordot(T, self.tok_all, dims=([0, -1], [1, 2])).abs() # (self.emb.v)
        new_sample = torch.multinomial(probs, num_samples=1).item()
        out_T = torch.einsum(out_T, [0, 1, 2, 3, 4, 5], self.emb([new_sample]).squeeze(0), [0, 1], [2, 3, 4, 5])
        if site == self.n-1:
            out_T = out_T.view(-1).real # return real number for an exact probability
        return new_sample, out_T
    
    @torch.no_grad()
    def sample(self):
        out = []
        T = None
        for site in range(self.n):
            new_sample, T = self.sample_step(T, site)
            out.append(new_sample)
        return out, T # return sampled token & corresponding prob
        
def grad_proj_iso(X, dX):
    with torch.no_grad():
        mps_shape = X.shape
        iso_shape = X.view(-1, X.shape[-1]).shape
        X, dX = X.view(iso_shape), dX.view(iso_shape)
        G = dX - 0.5 * (X @ X.t().conj() @ dX + X @ dX.t().conj() @ X)
        #A = dX @ X.t().conj() - X @ dX.t().conj() + 0.5 * X @ (dX.t().conj() @ X - X.t().conj() @ dX) @ X.t().conj()
    return G.reshape(mps_shape)#, A.reshape(mps_shape)

def iso_op(optimizer, loss):
    for group in optimizer.param_groups:
        # isometric constrain
        if group["name"] == "iso":
            for p in group["params"]:
                G = grad_proj_iso(p, p.grad)
                p.grad = 0.5 * G
        # quantum measure constrain
        #if group["name"] == "emb":
        #    for p in group["params"]:
        #        G = grad_proj_emb(p, p.grad)
        #        p.grad = 0.5 * G