
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

from Lpips import VGG16LPIPS
from Discriminator import Discriminator
from util import save_vae, load_vae
from torchmetrics.image.fid import FrechetInceptionDistance


class trainer_VQ():
    def __init__(self, device, n_epochs, lr, model, training_vae_continue, model_path=None):
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.training_continue = training_vae_continue
        self.model = model.to(device)
        self.perceptual_loss_fn = VGG16LPIPS().to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr,amsgrad=False)

        
        if training_vae_continue:
            load_vae(self.model, self.optimizer, model_path)

        
    def train(self, train_dataloader, model_name):
        for epoch in range(self.n_epochs):
            train_loss = 0.0
            recon_loss_sum = 0.0
            per_loss_sum = 0.0
            self.model.train()
            for i, batch in enumerate(tqdm(train_dataloader)):
                # forward
                
                x = batch['pixel_values']

                x = x.to(self.device)
                reconstructed, _, recon_loss  = self.model(x)

                #perceptual loss
                per_loss = self.perceptual_loss_fn(x, reconstructed)

                gen_base_loss = recon_loss + per_loss

                total_gen_loss = gen_base_loss
               

                # Backprop for Generator
                self.optimizer.zero_grad()
                total_gen_loss.backward()
                self.optimizer.step()
                
                # 로그 저장
                train_loss += total_gen_loss.item()
                recon_loss_sum += recon_loss.item()
                per_loss_sum += per_loss.item()

                        # Epoch별 로그 출력
            
            self.model.eval()
            batch = next(iter(train_dataloader))
        
            real_images = batch["pixel_values"].to(self.device)[:10]
            real_images_01 = real_images
            with torch.no_grad():
                fake_images_01, _, _ = self.model(real_images)
                fake_images_01 = fake_images_01
            fid_metric = FrechetInceptionDistance(normalize=True, input_img_size=(3,256,256)).to(self.device)

            # real, fake 이미지 업데이트
            fid_metric.update(real_images_01, real=True)
            fid_metric.update(fake_images_01, real=False)

            # 5) FID 계산
            fid_score = fid_metric.compute().item()

            fid_log_file = "VAE_fid_scores.txt"
            with open(fid_log_file, "a") as f:
                f.write(f"{epoch} FID Score: {fid_score}\n")
            
            dataset_size = len(train_dataloader.dataset)
            print(f"===> Epoch: {epoch+1} "
                  f"Generator Loss: {train_loss/dataset_size} "
                  f"L2 Loss: {recon_loss_sum/dataset_size} "
                  f"Perceptual Loss: {per_loss_sum/dataset_size} "
                  f"FID: {fid_score} ")

            save_vae(self.model, self.optimizer, model_name)

    def test(self, x):
        x = x.to(self.device)
        x_reconst, _, _ = self.model(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x_reconst = x_reconst.permute(0, 2, 3, 1).contiguous()


        fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2행 num_images열

        # 원본 이미지 출력
        for i in range(5):
            axes[0, i].imshow(x.cpu()[i]*0.5+0.5)
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')

        # 재구성된 이미지 출력
        for i in range(5):
            axes[1, i].imshow(x_reconst.cpu().detach().numpy()[i]*0.5+0.5)
            axes[1, i].set_title(f"Reconstructed {i+1}")
            axes[1, i].axis('off')

        # 레이아웃 조정 및 출력
        plt.tight_layout()
        plt.show()


