import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16LPIPS(nn.Module):
    """
    VGG16의 몇 개 레이어에서 추출한 feature 맵을 이용해
    입력 두 이미지의 Perceptual Loss를 구합니다.
    """
    def __init__(self, resize=True):
        super(VGG16LPIPS, self).__init__()
        # 사전 학습된 VGG16 로드
        vgg_pretrained = models.vgg16(pretrained=True).features
        
        # 필요한 레이어만 추출해서 nn.Sequential로 구성
        # conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 등 원하는 레이어까지 자유롭게 설정 가능
        self.slice1 = nn.Sequential(*[vgg_pretrained[i] for i in range(4)])   # relu1_2까지
        self.slice2 = nn.Sequential(*[vgg_pretrained[i] for i in range(4, 9)])  # relu2_2까지
        self.slice3 = nn.Sequential(*[vgg_pretrained[i] for i in range(9, 16)]) # relu3_3까지
        self.slice4 = nn.Sequential(*[vgg_pretrained[i] for i in range(16, 23)])# relu4_3까지
        self.slice5 = nn.Sequential(*[vgg_pretrained[i] for i in range(23, 30)])# relu5_3까지
        # 필요하면 slice5도 구성 가능
        
        # VGG16 파라미터는 학습되지 않도록 고정
        for param in self.parameters():
            param.requires_grad = False
        
        # 입력 이미지를 VGG16의 학습 시 입력 스케일에 맞추기 위해 사용
        # 여기서는 Imagenet 통계치에 맞게 정규화하는 과정을 forward에서 수행
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """
        x, y: [B, 3, H, W] 범위의 텐서(0~1 or 0~255, 현재 코드는 0~1 가정)
        두 이미지 x, y의 feature perceptual loss를 계산 후 스칼라 반환.
        """
        # 1) VGG16 입력 스케일로 맞추기 위해 정규화
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        # 2) 각 slice 별로 feature를 구함
        x1 = self.slice1(x)
        x2 = self.slice2(x1)
        x3 = self.slice3(x2)
        x4 = self.slice4(x3)
        x5 = self.slice5(x4)
        
        y1 = self.slice1(y)
        y2 = self.slice2(y1)
        y3 = self.slice3(y2)
        y4 = self.slice4(y3)
        y5 = self.slice5(y4)
        
        # 3) 여러 레이어에서의 feature 차이를 계산 (L2 distance 사용 예시)
        loss = 0
        loss += F.mse_loss(x1, y1)
        loss += F.mse_loss(x2, y2)
        loss += F.mse_loss(x3, y3)
        loss += F.mse_loss(x4, y4)
        loss += F.mse_loss(x5, y5)
        
        return loss/5


