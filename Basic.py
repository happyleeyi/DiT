from inspect import isfunction
from einops.layers.torch import Rearrange
from torch import nn, einsum


## 존재하면 1 존재하지 않으면 0
def exists(x):
    return x is not None

## val이 존재하면 val return 아니면 d return
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

## num을 divisor씩 나누어 arr에 넣어줌 예를 들어 num=22, divisor=6이면 arr는 [6,6,6,4]
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0 :
        arr.append(remainder)
    return arr

## residual block
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn=fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

## upsampling(여기선 2배 크기로 가까운 픽셀 복사하여 늘림) 후 conv2d
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

## downsampling(여기선 이미지 가로 세로를 1/2로 줄이고 그 두께를 4배시킴) 후 1x1 conv2d
def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)