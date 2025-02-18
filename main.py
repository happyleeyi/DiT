import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose

from Model import DiT
from Trainer import Trainer
from Trainer_VAE import trainer_VQ
from Model_VAE import VQVAE
from util import extract_features

import kagglehub

# Download latest version
#path = kagglehub.dataset_download("crawford/cat-dataset")

#print("Path to dataset files:", path)

# load dataset from the hub
# catface : ./cat_face/cat_face
# celeb : tonyassi/celebrity-1000      #25

dataset = load_dataset(path="tonyassi/celebrity-1000")
trial = 164
model_name = "./celeb_dit_1000," + str(trial) + ".pt"
VAE_name = "./celeb_VQVAE_8.pt"
feature_path = "./celeb_feature.npy"
feature_trained = True
image_size = 256
latent_size = int(image_size/8)
embedding_dim = 4
channels = 3
batch_size = 8
timesteps = 1000      #총 time step
lr = 1e-4
lr_vae = 1e-4
epochs_vae = 30
epochs = 100

training_vae_continue = True
testing_vae = False            #vae testing 할거면 1
training_state_vae = False     #training 해야되면 1, 모델있으면 0
z_channels = 4

training_state = 0       #training 단계면 1 sampling 단계면 0
sample_time = 100
save_folder = './XL, 2, 170, 8배/'
check = 1                  #sampling 처음이면 1, 아니면 0
dt = 100                    #ddim time step 몇번씩 건너뛸지
gpu = 1                    #gpu 쓸지
repeat = None              #반복해서 sampling 할지
device = "cuda" if torch.cuda.is_available() else "cpu"

dim = 1152
patch_size = 2
depth = 28
num_heads = 16
mlp_dim = 1152*4


vae_down_channels=[64, 128, 256, 256]
vae_mid_channels=[256, 256]
vae_down_sample=[True, True, True]
vae_attns = [False, False, False]
vae_num_down_layers = 2
vae_num_mid_layers = 2
vae_num_up_layers = 2
vae_z_channels = 4
vae_codebook_size = 8192
vae_norm_channels = 32
vae_num_heads = 4


transform = Compose([
            #transforms.CenterCrop(256),
            transforms.Resize((image_size,image_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t*2) - 1),
])

# define function 

def transforms(examples):   

   examples["pixel_values"] = [transform(image) for image in examples["image"]]

   del examples["image"]

   return examples

transformed_dataset = dataset.with_transform(transforms)

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=12, shuffle=True)
""" dataiter = iter(dataloader)
plt.imshow(np.fliplr(np.rot90(np.transpose(next(dataiter)['pixel_values'][0]/2+0.5), 3)))
plt.show() """

vae = VQVAE(channels, vae_down_channels, vae_mid_channels, vae_down_sample, vae_attns, vae_num_down_layers, vae_num_mid_layers,
             vae_num_up_layers, vae_z_channels, vae_codebook_size, vae_norm_channels, vae_num_heads)

dit = DiT(latent_size, dim, patch_size, depth, num_heads, mlp_dim, z_channels)


Trainer_vq = trainer_VQ(device, epochs_vae, lr_vae, vae, training_vae_continue, VAE_name)
if training_state_vae:
   Trainer_vq.train(dataloader, VAE_name)
if testing_vae:
   data_iter = iter(dataloader)
   batch = next(data_iter)  # 첫 번째 배치 가져오기

   images = batch["pixel_values"]  # 이미지 데이터 추출

   # 첫 번째 이미지 선택
   first_image = images[0:5]  # 첫 번째 이미지
   print(first_image.shape)
   Trainer_vq.test(first_image)

if feature_trained is False:
   extract_features(device, vae, dataloader, feature_path)


features_np = np.load(feature_path)
features_tensor = torch.from_numpy(features_np)
feature_dataset = TensorDataset(features_tensor)
feature_dataloader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)


repeat = int(timesteps/dt)

t = timesteps -1

if training_state:
   if trial == 0:
      model_name = None
   trainer = Trainer(device, feature_dataloader, latent_size, embedding_dim, batch_size, timesteps, lr, model_name, vae, dit)
   for epoch in range(epochs):
      print(str(trial+epoch+1)+" training start")
      if epoch == 0:
         load_ckpt = model_name
      else :
         load_ckpt = "./celeb_dit_1000," + str(trial+epoch) + ".pt"
      save_ckpt = "./celeb_dit_1000," + str(trial+epoch+1) + ".pt"
      
      trainer.training(load_ckpt, save_ckpt, 1)
      trainer.save_sample(dt, 1, "./", False)
      trainer.fid_one_batch(trial+epoch+1, dt, dataloader)
else:
   trainer = Trainer(device, feature_dataloader, latent_size, embedding_dim, batch_size, timesteps, lr, model_name, vae, dit)
   trainer.save_sample(dt, sample_time, save_folder)
   
      