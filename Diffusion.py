import torch
import numpy as np

import torch.nn.functional as F



class Diffusion():
    def __init__(self, timesteps, beta_start, beta_end):
        self.timesteps = timesteps

        self.betas = torch.linspace(beta_start, beta_end, self.timesteps)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
    
    def p_sample(self, model, x, t, dt, t_index):
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t_pred = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t-dt, x.shape
        )

        sqrt_alphas_cumprod_t_pred = self.extract(self.sqrt_alphas_cumprod, t-dt, x.shape)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x.shape)
        
        # Use our model (noise predictor) to predict the mean

        model_mean = sqrt_alphas_cumprod_t_pred * (
            (x - sqrt_one_minus_alphas_cumprod_t * model(x, t)) / sqrt_alphas_cumprod_t
        ) + sqrt_one_minus_alphas_cumprod_t_pred * model(x, t)

        if t_index == 0:
            return model_mean
        else:
            #posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            #noise = torch.randn_like(x)

            return model_mean

    def p_sample_loop(self, check, t, dt, model, shape):
        device = next(model.parameters()).device

        b = shape[0]
        num=0
        # start from pure noise (for each example in the batch)
        if check == 1:
            img = torch.randn(shape, device=device)
            imgs = []
        else:
            imgs = np.load('samples.npy')
            imgs = imgs.tolist()
            img = torch.tensor(imgs[-1], device=device)
            num = len(imgs)

        img = self.p_sample(model, img, torch.full((b,), t, device=device, dtype=torch.long), dt, t)
        imgs.append(img.cpu().detach().numpy())

        return imgs
        
    def sample(self, check, t, dt, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(check, t,  dt, model, shape=(batch_size, channels, image_size, image_size))