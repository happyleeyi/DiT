import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel

class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, bias: bool = False, is_causal: bool = False, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, "Channel must be divisible by num_heads"

        self.num_heads = num_heads
        self.channel = channels
        self.head_dim = channels // num_heads
        self.is_causal = is_causal

        # Linear layers for query, key, value projections
        self.qkv_proj = nn.Linear(channels, 3 * channels, bias=bias)
        # Output projection
        self.out_proj = nn.Linear(channels, channels, bias=bias)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch size, Channel, H, W)
        batch_size, N, channel = x.shape

        # Compute query, key, value
        qkv = self.qkv_proj(x)  # Shape: (Batch size, H * W, 3 * Channel)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (Batch size, num_heads, H * W, head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)      # (Batch size, num_heads, H * W, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (Batch size, num_heads, H * W, head_dim)

        # Perform scaled dot-product attention
        attention_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=self.dropout.p, is_causal=self.is_causal
            )  # Shape: (Batch size, num_heads, H * W, head_dim)

        # Reshape back to (Batch size, H * W, Channel)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, N, channel)

        # Apply output projection
        output = self.out_proj(attention_output)  # Shape: (Batch size, H * W, Channel)

        return output

class FlashAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, bias: bool = False, is_causal: bool = False, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, "Channel must be divisible by num_heads"

        self.num_heads = num_heads
        self.channel = channels
        self.head_dim = channels // num_heads
        self.is_causal = is_causal

        # Linear layers for query, key, value projections
        self.qkv_proj = nn.Linear(channels, 3 * channels, bias=bias)
        # Output projection
        self.out_proj = nn.Linear(channels, channels, bias=bias)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch size, Channel, H, W)
        batch_size, N, channel = x.shape

        # Compute query, key, value
        qkv = self.qkv_proj(x)  # Shape: (Batch size, H * W, 3 * Channel)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim)  # (Batch size, H * W, num_heads, head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)      # (Batch size, H * W, num_heads, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)  # (Batch size, H * W, num_heads, head_dim)

        # Perform scaled dot-product attention
        attention_output = flash_attn_func(
                query, key, value, softmax_scale=1/math.sqrt(self.head_dim)
            )  # Shape: (Batch size, H * W, num_heads, head_dim)

        # Reshape back to (Batch size, H * W, Channel)
        attention_output = attention_output.reshape(batch_size, N, channel)

        # Apply output projection
        output = self.out_proj(attention_output)  # Shape: (Batch size, H * W, Channel)

        return output
    
class FlashSelfAttention(nn.Module):
    def __init__(self, channels, num_heads, dropout_p=0.0, causal=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv_proj = nn.Linear(channels, 3 * channels)
        self.o_proj = nn.Linear(channels, channels)

        # 추가한 부분
        self.layernorm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout_p)

        self.dropout_p = dropout_p
        self.causal = causal

    def forward(self, x):
        # x: (batch_size, seq_len, channels)

        # [1] Residual connection을 위한 원본을 저장
        residual = x

        # [2] LayerNorm 적용
        x = self.layernorm(x)

        batch_size, seq_len, _ = x.size()

        # [3] Q, K, V 계산
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * channels)
        # (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # [4] Flash Attention 수행 (스케일링 추가)
        out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),  # \sqrt{d_head} 스케일링
            causal=self.causal
        )  # (batch_size, seq_len, num_heads, head_dim)

        # [5] (batch_size, seq_len, channels) 형태로 복원 후 최종 프로젝트
        out = out.view(batch_size, seq_len, self.channels)
        out = self.o_proj(out)

        # [6] Residual + Dropout
        out = residual + self.dropout(out)

        return out

class LinearAttentionBlock(nn.Module):
    """
    입력: [B, C, H, W]
    Linear Attention의 입력: [Batch, Seq_len, Embed_dim] 형태
    여기서는 Seq_len = H*W, Embed_dim = C 로 설정
    """
    def __init__(self, channels, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.groupnorm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        x_reshape = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)

        # Query, Key, Value projections
        Q = self.q_proj(x_reshape)  # (B, HW, C)
        K = self.k_proj(x_reshape)  # (B, HW, C)
        V = self.v_proj(x_reshape)  # (B, HW, C)

        # Apply softmax to Q
        Q = F.softmax(Q, dim=1)  # (B, HW, C)

        # Apply softmax over the sequence length dimension (HW) for K
        K = F.softmax(K, dim=1)  # (B, HW, C)

        # Compute Linear Attention
        KV = torch.bmm(K.transpose(1, 2), V)  # (B, C, C)
        Z = torch.bmm(Q, KV)  # (B, HW, C)

        # Residual connection
        x_reshape = x_reshape + Z

        # Group Normalization
        x_reshape = self.groupnorm(x_reshape.permute(0, 2, 1)).permute(0, 2, 1)  # (B, HW, C)

        # (B, HW, C) -> (B, C, H, W)
        out = x_reshape.permute(0, 2, 1).view(B, C, H, W)
        return out
        
class LinearAttention(nn.Module):
    """
    간단한 Linear Attention 구현 예시.
    """
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0, (
            "channels must be divisible by num_heads."
        )
        
        self.head_dim = channels // num_heads
        
        # Q, K, V를 구하기 위한 projection 레이어
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        
        # Multi-Head Attention 출력을 합친 뒤 최종적으로 다시 projection
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        """
        x.shape = (batch_size, seq_len, channels)
        """
        bsz, seq_len, _ = x.shape
        
        # 1) Q, K, V 계산
        q = self.q_proj(x)  # (bsz, seq_len, channels)
        k = self.k_proj(x)  # (bsz, seq_len, channels)
        v = self.v_proj(x)  # (bsz, seq_len, channels)

        # 2) (B, seq_len, channels) -> (B, num_heads, seq_len, head_dim) 로 shape 변경
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) 커널(피쳐 맵) 적용. phi(Q), phi(K)
        q_prime = F.softmax(q)  # (bsz, num_heads, seq_len, head_dim)
        k_prime = F.softmax(k)  # (bsz, num_heads, seq_len, head_dim)

        # 4) Linear Attention 식
        #    numerator = Q' @ (K'^T @ V)
        #    denominator = 1 / (Q' @ K'^T)
        #    (여기서는 batch, head 차원을 분리해 생각)

        # K'^T @ V -> seq_len x head_dim 과 seq_len x head_dim를 행렬곱 하려면
        # 먼저 K', V에 대해 seq_len 차원을 맨 앞으로 가져와서 곱함
        # 다만, PyTorch에서 이를 효율적으로 계산하기 위해 B, H, head_dim, seq_len 형태 변경 후 곱하면 편할 수 있음.
        
        # (bsz, num_heads, seq_len, head_dim) -> (bsz, num_heads, head_dim, seq_len)
        k_prime_t = k_prime.transpose(-1, -2)  # (bsz, num_heads, head_dim, seq_len)
        
        # K'^T @ V: (bsz, num_heads, head_dim, seq_len) @ (bsz, num_heads, seq_len, head_dim)
        # = (bsz, num_heads, head_dim, head_dim)
        kv = torch.matmul(k_prime_t, v)

        # Q' @ (K'^T @ V): (bsz, num_heads, seq_len, head_dim) @ (bsz, num_heads, head_dim, head_dim)
        # = (bsz, num_heads, seq_len, head_dim)
        out_num = torch.matmul(q_prime, kv)

        # Q' @ K'^T: (bsz, num_heads, seq_len, head_dim) @ (bsz, num_heads, head_dim, seq_len)
        # = (bsz, num_heads, seq_len, seq_len)
        # 여기서 각 위치별(normalization)로 나눠야 하므로, 
        # denominator(= Q'K'^T)를 seq_len별로 합산한 뒤 broadcasting 형태로 나누는 방법을 사용
        # 그러나 나머지 구현 방식에 따라 다를 수 있음 (ex. cumulative sum)
        
        k_prime_sum = k_prime.sum(dim=2)  # (bsz, num_heads, head_dim)
        # shape 상 seq_len 차원만 제외하고 합친 값을 Q'와 곱할 때
        # seq_len 위치마다 K'를 모두 합친 것과의 dot를 하게 됨
        
        # (bsz, num_heads, seq_len, head_dim)와 (bsz, num_heads, head_dim) -> (bsz, num_heads, seq_len)
        denominator = torch.einsum('bhld,bhd->bhl', q_prime, k_prime_sum) + 1e-6
        denominator = denominator.unsqueeze(-1)  # (bsz, num_heads, seq_len, 1)
        
        out = out_num / denominator  # (bsz, num_heads, seq_len, head_dim)

        # 5) 최종적으로 (B, seq_len, num_heads, head_dim)를 합쳐서 (B, seq_len, channels)로 변환
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.channels)

        # 6) 마지막 projection
        out = self.out_proj(out)

        return out
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels, cross_attn=False, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            
            self.attentions = nn.ModuleList(
                [LinearAttentionBlock(channels=out_channels, num_heads=num_heads)  # 중간에 한 번 적용
                 for _ in range(num_layers)]
            )
        
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            if self.attn:
                out = self.attentions[i](out)
            
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
            
        # Downsample
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    """
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels, cross_attn=None, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )

        self.attentions = nn.ModuleList(
            [LinearAttentionBlock(channels=out_channels, num_heads=num_heads)  # 중간에 한 번 적용
             for _ in range(num_layers)]
        )
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )
    
    def forward(self, x, t_emb=None, context=None):
        out = x
        
        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            # Attention Block
            out = self.attentions[i](out)
            
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
                
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        
        return out

class UpBlock(nn.Module):
    """
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            
            self.attentions = nn.ModuleList(
                [LinearAttentionBlock(channels=out_channels, num_heads=num_heads)  # 중간에 한 번 적용
                 for _ in range(num_layers)]
            )
            
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1) if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down=None, t_emb=None):
        # Upsample
        x = self.up_sample_conv(x)
        
        # Concat with Downblock output
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Self Attention
            if self.attn:
                out = self.attentions[i](out)
        return out
    
class UpBlockUnet(nn.Module):
    """
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            
            self.attentions = nn.ModuleList(
                [LinearAttentionBlock(channels=out_channels, num_heads=num_heads)  # 중간에 한 번 적용
                 for _ in range(num_layers)]
            )
            
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1) if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down=None, t_emb=None):
        # Upsample
        x = self.up_sample_conv(x)
        
        # Concat with Downblock output
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Self Attention
            if self.attn:
                out = self.attentions[i](out)
        return out