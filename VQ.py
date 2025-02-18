import torch.nn as nn
import torch
import torch.nn.functional as F

class VQ(nn.Module):
    
    def __init__(self,num_embeddings=512,embedding_dim=64,commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(self.num_embeddings,self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings,1/self.num_embeddings)
    
    def forward(self,inputs):
        input_shape = inputs.shape
        N, D, H, W = input_shape

        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous()
        
        # permute 후 shape: (N, H, W, D)
        flat_inputs = flat_inputs.view(-1, D)  # -> (N*H*W, D)
        
        distances = torch.cdist(flat_inputs,self.embeddings.weight)
        encoding_index = torch.argmin(distances,dim=1) 
        
        quantized = torch.index_select(self.embeddings.weight,0,encoding_index)
        
        quantized = quantized.view(N, H, W, D)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        e_latent_loss = F.mse_loss(quantized.detach(),inputs)
        q_latent_loss = F.mse_loss(quantized,inputs.detach())
        c_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()

        return c_loss, quantized