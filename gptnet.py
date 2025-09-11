import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from typing import List
'''----------------------------------------------------------------------------------------'''

@dataclass
class GPTconfig: # Use this as an input file to configure the architecture of the GPT model
    context_length = 1024 #maximum context length (number of tokens) the model can handle at a time
    blocks = 2 # number of transformer blocks to be included in the GPT
    num_heads = 2 # number of heads for multi-headed attention
    h_dim = 64 # the embedding dimension for each token
    lin_dim = 64 # number of neurons to be included in the linear layer of each transformer
    vocab_size = 50257 # number of tokens in the vocabulary (50,000 BPE merges + 256 UTF-8 byte tokens + 1 End of Text token)
    
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTconfig) -> None:
        super().__init__()
        self.T = config.context_length
        self.n_heads = config.num_heads
        self.h_dim = config.h_dim
        
        self.qkv_lin = nn.Linear(config.h_dim,3*config.h_dim)
        self.register_buffer('triller',torch.tril(torch.ones(1,1,self.T,self.T)))
        self.proj = nn.Linear(config.h_dim,config.h_dim)
        
    def forward(self,x):
        B,T,C = x.shape
        qkv_val = self.qkv_lin(x)
        q,k,v = qkv_val.split(self.h_dim,dim=2)
        q = q.view(B,T, self.n_heads, C // self.n_heads).transpose(1,2)
        k = k.view(B,T, self.n_heads, C // self.n_heads).transpose(1,2)
        v = v.view(B,T, self.n_heads, C // self.n_heads).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.shape[-1]))
        att = att.masked_fill(self.triller[:,:,:T,:T]==0,float('-inf'))
        att = F.softmax(att,dim=-1)
        o = att @ v
        o = o.transpose(1,2).contiguous().view(B,T,C)
        o = self.proj(o)
        return o


class MLP(nn.Module):
    def __init__(self, config: GPTconfig) -> None:
        super().__init__()
        self.lin1 = nn.Linear(config.h_dim,config.lin_dim)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(config.lin_dim,config.h_dim)
        
    def forward(self,x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTconfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.h_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.h_dim)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTconfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(self.config.vocab_size,self.config.h_dim) # Convert each token to the given h-dim embeddings
        self.pte = nn.Embedding(self.config.context_length,self.config.h_dim) # Calculate Positional Embedding for each token based on the index location
        self.transformer_blocks = nn.ModuleList([Block(self.config) for _ in range(config.blocks)]) # Creating the Transformer blocks as a list
        self.hidden_transformer_layers = nn.Sequential(*self.transformer_blocks) # unpacking the blocks as a sequential executor
        self.ln_forward = nn.LayerNorm(config.h_dim) # final layer noramlisation layer
        self.lm_head = nn.Linear(config.h_dim, config.vocab_size, bias=False) # Prediction of next token from the vocab
        
    def forward(self,x):
        x_emb = self.wte(x)
        x_pos =  self.pte(torch.tensor([i for i in range(x.shape[1])])) # Building the position embedding from the length of the tokens from shape (B,T)
        x = x_emb + x_pos
        x = self.hidden_transformer_layers(x)
        x = self.ln_forward(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self,idx: torch.Tensor, max_new_gens: int = 1, temperature: float = 1.0) -> List[int]:
        for _ in range(max_new_gens):
            idx_s = idx if idx.shape[1] <= self.config.context_length else idx[:,:-self.config.context_length]
            logits = self.forward(idx_s)
            last_token = logits[:,-1,:] / temperature
            probs = F.softmax(last_token,dim=-1) 
            sample = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,sample),dim=1)
        return idx.view(-1).tolist()
    
    def calculate_number_of_params(self, only_trainable: bool = False, non_embedding: bool = False) -> int:
        
        if only_trainable:
            param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
        if non_embedding:
            param_count -= self.wte.weight.numel()
            
        return param_count
