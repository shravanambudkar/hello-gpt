import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
'''----------------------------------------------------------------------------------------'''

@dataclass
class GPTconfig:
    token_length = 1024
    blocks = 2
    num_heads = 2
    h_dim = 64
    n_dim = 64
    vocab_size = 50257
    


class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.T = config.token_length
        self.n_heads = config.num_heads
        self.h_dim = config.h_dim
        
        self.qkv_lin = nn.Linear(config.h_dim,3*config.h_dim)
        self.register_buffer('triller',torch.tril(torch.ones(1,1,self.T,self.T)))
        self.proj = nn.Linear(config.h_dim,config.h_dim)
        
    def forward(self,x):
        B,T,C = x.shape
        qkv_val = self.qkv_lin(x)
        q,k,v = qkv_val.split(self.h_dim,dim=2)
        q_reproj = q.view(B,T, self.n_heads, C // self.n_heads).transpose(1,2)
        k_reproj = k.view(B,T, self.n_heads, C // self.n_heads).transpose(1,2)
        v_reproj = v.view(B,T, self.n_heads, C // self.n_heads).transpose(1,2)
        att = (q_reproj @ k_reproj.transpose(-2,-1)) * (1.0/math.sqrt(k_reproj.shape[-1]))
        att = att.masked_fill(self.triller[:,:,:T,:T]==0,float('-inf'))
        att = F.softmax(att,dim=-1)
        o = att @ v_reproj
        o = o.transpose(1,2).contiguous().view(B,T,C)
        o = self.proj(o)
        return o
        
        
class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lin1 = nn.Linear(config.h_dim,config.n_dim)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(config.n_dim,config.h_dim)
        
    def forward(self,x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x

class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.h_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.h_dim)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT2(nn.Module):
    def __init__(self, config: GPTconfig) -> None:
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size,self.config.h_dim),
            pte = nn.Embedding(self.config.token_length,self.config.h_dim),
            h = nn.ModuleList([Block(self.config) for _ in range(config.blocks)]),
            ln_f = nn.LayerNorm(config.h_dim)
        )
        )
        
        self.lm_head = nn.Linear(config.h_dim, config.vocab_size, bias=False)
        
    def forward(self,x):
        x_emb = self.transformer.wte(x)
        x_pos =  self.transformer.pte(torch.tensor([i for i in range(x.shape[1])]))
        x = x_emb + x_pos
        for b in self.transformer.h:
            x = b(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self,idx):
        logits = self.forward(idx)
        last_token = logits[:,-1,:]
        probs = F.softmax(last_token,dim=-1)
        sample = torch.multinomial(probs,num_samples=1)
        return torch.cat((idx,sample),dim=1).view(-1).tolist()
    

'''----------------------------------------------------------------------------------------'''

from data_proc import BatchLoader
import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-2")

def main():
    loader = BatchLoader(batch_size=1,context_length=1024)
    X, Y = loader.next_batch()
    #initialize model
    model = GPT2(GPTconfig())
    # logits = model(X)
    new_idx = model.generate(torch.tensor([[5962, 22307, 25, 198]]))
    print(new_idx)
    print(tokenizer.decode(new_idx))

if __name__ == '__main__':
    main()
