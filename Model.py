import torch
from torch import nn


from Blocks import SpatialSelfAttention, LinearAttention, FlashSelfAttention, FlashAttention

## Position embedding
def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb
class AdaLayerNorm(nn.Module):
    """
    LayerNorm에 조건을 이용한 scale/shift(γ, β)를 더해주는 모듈.
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        # 기본적인 LayerNorm 파라미터
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)
        
        # 여기서는 γ, β를 직접 학습하지 않고
        # 외부에서 MLP 등의 조건 네트워크가 예측한 파라미터를 입력받아 적용
        # (elementwise_affine=False 로 설정했으므로 기본 scale/shift는 없음)
        
    def forward(self, x, gamma_beta=None):
        # 먼저 기본 LayerNorm
        x = self.norm(x)
        
        if gamma_beta is not None:
            # gamma_beta: (B, 2*dim) 이라고 가정
            # 앞 절반은 gamma, 뒷 절반은 beta
            B, D = gamma_beta.shape
            # gamma_beta.shape = (B, 2*D) 라 가정
            gamma, beta = gamma_beta[:, :D//2], gamma_beta[:, D//2:]
            
            # 브로드캐스트를 위해 (B, 1, D) 형태로 reshape
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            
            # scale/shift
            x = x * (1 + gamma) + beta
        
        return x

class DiTBlock(nn.Module):
    """
    DiT Block with adaLN-Zero.
    """
    def __init__(self, dim, num_heads, mlp_dim, rate = 0.0):
        super().__init__()
        self.dim = dim
        
        self.norm1 = AdaLayerNorm(dim)  # 첫 번째 LayerNorm
        self.attn = SpatialSelfAttention(dim, num_heads)
        self.norm2 = AdaLayerNorm(dim)  # 두 번째 LayerNorm

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate),
        )
        
        # 조건(예: timestep, label 등)을 받아서 gamma/beta를 만들기 위한 간단 MLP
        # 실제 구현에서는 timestep + label 임베딩 등을 concatenation해서 처리할 수 있음
        self.condition_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),  # 예: dim차원 조건 -> (γ1, β1, γ2, β2)
        )
        
        # skip 연결에 곱할 alpha. 
        # 그림에서 알파 기호(α1, α2)에 해당하지만, 간단히 1.0으로 놓을 수도 있음
        self.alpha1 = nn.Linear(dim, dim)
        self.alpha2 = nn.Linear(dim, dim)
        
    def forward(self, x, cond):
        """
        cond: (B, cond_dim) - 예: timestep, label 등의 임베딩
        """
        
        # 1) cond를 통해 γ1, β1, γ2, β2 추출
        #    여기서는 (B, 4*D) 형태로 나온다고 가정

        cond_out = self.condition_mlp(cond)  # (B, 4D)
        
        # 앞쪽 절반(2D)은 첫 번째 AdaLayerNorm용(γ1, β1), 
        # 뒤쪽 절반(2D)은 두 번째 AdaLayerNorm용(γ2, β2)
        ada1 = cond_out[:, :2*self.dim]   # (B, 2*D)
        ada2 = cond_out[:, 2*self.dim:]   # (B, 2*D)
        
        # 2) Multi-Head Self-Attention + Skip
        #    norm1(x)에서 γ1, β1 적용
        x_norm = self.norm1(x, gamma_beta=ada1)  # (B, N, D)
        attn_out = self.attn(x_norm)  # (B, N, D)
        x = x + self.alpha1(cond).unsqueeze(1) * attn_out  # skip connection
        
        # 3) Feed Forward + Skip
        x_norm = self.norm2(x, gamma_beta=ada2)  # (B, N, D)
        ff_out = self.mlp(x_norm)  # (B, N, D)
        x = x + self.alpha2(cond).unsqueeze(1) * ff_out
        
        return x
class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
         

    def forward(self, x, c):
        scale = self.gamma(c).unsqueeze(1)
        shift = self.beta(c).unsqueeze(1)
        x = self.ln_final(x) * (1+scale) + shift
        x = self.linear(x)
        return x
class DiT(nn.Module):
    def __init__(self, img_size, dim=64, patch_size=4, depth=3, heads=4, mlp_dim=512, in_channels=3):
        super(DiT, self).__init__()
        self.dim = dim
        self.n_patches = (img_size // patch_size)**2 
        self.depth = depth
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, dim))
        self.patches = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, 
                      stride=patch_size, padding=0, bias=False),
        )
        self.transformer = nn.ModuleList()
        for i in range(self.depth):
            self.transformer.append(
                DiTBlock(dim, heads, mlp_dim)
            )
        self.emb = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.final = FinalLayer(dim, patch_size, in_channels)
        self.ps = nn.PixelShuffle(patch_size)
    def forward(self, x, t):
        t = get_time_embedding(torch.as_tensor(t).long(), self.dim)
        t = self.emb(t)
        x = self.patches(x)
        B, C, H, W = x.shape
        x = x.permute([0, 2, 3, 1]).reshape([B, H * W, C])
        x += self.pos_embedding
        for layer in self.transformer:
            x = layer(x, t)

        x = self.final(x, t).permute([0, 2, 1])
        x = x.reshape([B, -1, H, W])
        x = self.ps(x)
        return x
