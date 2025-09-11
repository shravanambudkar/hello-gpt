from tokenloader import BatchLoader
from gptnet import GPTconfig, GPT
import torch.nn.functional as F
from torch.optim import AdamW
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}')

torch.set_float32_matmul_precision('high')
train_batches = 500
batch_loader = BatchLoader(batch_size=4,context_length=1024)
model = GPT(GPTconfig())
print(f'number of parameters: {model.calculate_number_of_params(non_embedding=True)}')
model = model.to(device)
optimizer = AdamW(model.parameters(),lr=3e-4)
for _ in range(train_batches):
    optimizer.zero_grad()
    x, y = batch_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    with torch.autocast(device_type=device,dtype=torch.bfloat16):    
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T,-1),y.view(-1))
    print(f'current loss: {loss.item()}')
    loss.backward()
    optimizer.step()
    
