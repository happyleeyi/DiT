import torch
import numpy as np
import matplotlib.pyplot as plt

from torchmetrics.image.fid import FrechetInceptionDistance

from torch.optim import Adam

from tqdm.auto import tqdm

from Diffusion import Diffusion
from util import save_checkpoint, load_checkpoint, EMA


class Trainer():
    def __init__(self, device, dataloader, image_size, channels, batch_size, timesteps, lr, ckpt, vae, ldm):
        self.device = device
        self.dataloader = dataloader
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.model = ldm.to(self.device)
        self.lr = lr
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.ema = EMA(self.model, decay=0.999)
        if ckpt is not None:
            load_checkpoint(self.model, self.ema, self.optimizer, ckpt)
        self.diffusion = Diffusion(self.timesteps,0.0001,0.02)
        self.vae = vae

    def training(self, load_ckpt, save_ckpt, epochs):
        self.model.to(self.device)

        scaler = torch.amp.GradScaler('cuda')

        self.model.train()

        for epoch in range(epochs):
            loss_sum=0
            for step, batch in enumerate(tqdm(self.dataloader)):
                self.optimizer.zero_grad()

                batch_size = batch[0].shape[0]
                batch = batch[0].to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                with torch.amp.autocast('cuda'):
                    loss = self.diffusion.p_losses(self.model, batch, t, loss_type="huber")

                loss_sum+=loss.item()

                if step % 100 == 0:
                    print("Epoch, Loss:", epoch, loss_sum/100)
                    loss_sum=0

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                #loss.backward()
                #optimizer.step()
                self.ema.update()
        save_checkpoint(self.model, self.ema, self.optimizer, save_ckpt)
    def fid_one_batch(self, num, dt, real_dataloader):
        """
        ckpt_path: 확인하고자 하는 model의 checkpoint 경로.
                   내부에는 model, ema, optimizer state dict가 포함되어 있음.
        """
        # 1) 모델 상태 동기화 (체크포인트 로드)
        self.model.eval()

        # 2) Dataloader에서 단일 batch 추출
        #    여러 batch 평균이 아닌, 여기서는 예시로 첫 batch만 사용
        batch = next(iter(real_dataloader))
        
        real_images = batch["pixel_values"].to(self.device)[:10]
        real_images_01 = (real_images / 2) + 0.5

        # 3) 모델에서 생성된 이미지를 얻어오기 (Diffusion reverse + VAE decode 등)
        #    여기서는 간단 예시로 batch_size만큼 샘플링
        fake = []
        with torch.no_grad():
            # Diffusion Reverse 단계를 수행해서 fake latent를 얻는다 가정
            fake = self.save_sample(dt, real_images_01.shape[0], './', True, True)

            fake_images_01 = torch.tensor(np.array(fake)).to(self.device)/2+0.5

        # 4) torchmetrics의 FrechetInceptionDistance 사용
        #    normalize=True => 입력이 [0,1] 범위라면 내부적으로 [-1,1]로 바꿔 Inception 입력에 맞춤
        fid_metric = FrechetInceptionDistance(normalize=True).to(self.device)

        # real, fake 이미지 업데이트
        fid_metric.update(real_images_01, real=True)
        fid_metric.update(fake_images_01, real=False)

        # 5) FID 계산
        fid_score = fid_metric.compute().item()

        fid_log_file = "fid_scores.txt"
        with open(fid_log_file, "a") as f:
            f.write(f"{num} FID Score: {fid_score}\n")

        return fid_score
    def sampling(self, check, t, dt):
        self.model.to(self.device)
        self.ema.apply_shadow()
        self.model.eval()
        with torch.no_grad():
            samples = self.diffusion.sample(check, t, dt, self.model, image_size=self.image_size, batch_size=1, channels=self.channels)
        self.ema.restore()
        return samples
    def save_sample_time(self, check, t, dt, save_name=None):
        samples = self.sampling(check, t, dt)
        samples_img, _ = self.vae.quantize(torch.tensor(samples[-1]).to(self.device))
        samples_img = self.vae.decode(samples_img)
        samples_img = (samples_img[0]/2+0.5)
        np.save('samples.npy', np.array(samples))
        if save_name is not None:
            plt.imshow(samples_img.permute(1, 2, 0).cpu().detach().numpy())
            plt.savefig(save_name)
        #plt.show()
        return samples_img.cpu().detach().numpy()
    def save_sample(self, dt, sample_time, save_folder, sample_one_time = True, FID = False):
        repeat = int(self.timesteps/dt)
        sample_img = None
        sample_set = []
        for k in range(sample_time):
            t = self.timesteps -1
            check = 1
            save_name = save_folder + str(k)
            for i in range(repeat):
                print(t)
                if t<dt:
                    if FID:
                        sample_img = self.save_sample_time(check, t, dt-1)
                        sample_set.append(sample_img)
                    elif sample_one_time:
                        sample_img = self.save_sample_time(check, t, dt-1, save_name)
                    else:
                        sample_img = self.save_sample_time(check, t, dt-1, save_name+'fig_'+str(t))
                else:
                    if FID:
                        self.save_sample_time(check, t, dt)
                    elif sample_one_time:
                        self.save_sample_time(check, t, dt)
                    else:
                        self.save_sample_time(check, t, dt, save_name+'fig_'+str(t))
                check = 0
                t = t - dt
        return sample_set

