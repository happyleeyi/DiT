import torch
import numpy as np
from tqdm.auto import tqdm
def extract_features(device, vae, dataloader, save_path="features.npy"):
    """
    dataloader의 모든 이미지를 VAE로 encode하여
    그 latent feature들을 하나의 텐서로 모아 저장하는 메서드입니다.
    
    Args:
        save_path (str): 인코딩된 feature를 저장할 경로 (np.npy 등)
    """
    vae.eval()  # 혹시 모를 모델 상태 변경을 대비해 eval로 설정
    all_features = []

    # dataloader를 순회하며 각 배치를 인코딩
    for step, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
        images = batch["pixel_values"].to(device)
        with torch.no_grad():
            # VAE 인코딩
            latent = vae.encode(images)
        # CPU로 옮겨서 리스트에 모으기
        all_features.append(latent.detach().cpu())

    # 텐서로 합치기 (N, latent_dim, ...)
    all_features = torch.cat(all_features, dim=0)

    # 넘파이로 변환하여 파일로 저장
    np.save(save_path, all_features.numpy())
    print(f"Features saved at: {save_path}")
def save_vae(model, optimizer, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
def load_vae(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def save_checkpoint(model, ema, optimizer, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
def load_checkpoint(model, ema, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load EMA parameters if available
    if 'ema_state_dict' in checkpoint:
        for name, param in model.named_parameters():
            if param.requires_grad and name in checkpoint['ema_state_dict']:
                ema.shadow[name] = checkpoint['ema_state_dict'][name].clone()
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class EMA:
    def __init__(self, model, decay):
        """
        Initialize EMA class to manage exponential moving average of model parameters.
        
        Args:
            model (torch.nn.Module): The model for which EMA will track parameters.
            decay (float): Decay rate, typically a value close to 1, e.g., 0.999.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Store initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update shadow parameters with exponential decay.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Apply shadow (EMA) parameters to model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """
        Restore original model parameters from backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]