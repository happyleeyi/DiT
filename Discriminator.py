import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super(Discriminator, self).__init__()
        """
        간단한 CNN을 통한 real/fake 분류기
        최종적으로 single scalar를 출력한다고 가정
        """
        self.model = nn.Sequential(
            # [B, 3, H, W] -> [B, feature_dim, H/2, W/2]
            nn.Conv2d(in_channels, feature_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, feature_dim, H/2, W/2] -> [B, feature_dim*2, H/4, W/4]
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, feature_dim*2, H/4, W/4] -> [B, feature_dim*4, H/8, W/8]
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 글로벌 풀링 / 또는 Conv로 1x1까지 축소
            nn.Conv2d(feature_dim * 4, 1, 4, 1, 0)
        )

    def forward(self, x):
        """
        x: [B, 3, H, W]
        return: [B, 1] (real / fake 확률로 해석하기 위해 sigmoid를 적용하기 전의 logits)
        """
        out = self.model(x)  # [B, 1, n, n] (n은 입력 사이즈에 따라 달라짐)
        return out.view(-1, 1)  # [B, 1] 형태로 reshape
