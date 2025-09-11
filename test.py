'''
A simple script to test our hello-gpt execution
'''

from tokenloader import BatchLoader
import tiktoken
from gptnet import GPTconfig, GPT
import torch
tokenizer = tiktoken.encoding_for_model("gpt-2")

def main():
    loader = BatchLoader(batch_size=1,context_length=1024)
    X, Y = loader.next_batch()
    #initialize model
    model = GPT(GPTconfig())
    # logits = model(X)
    new_idx = model.generate(torch.tensor([[5962, 22307, 25, 198]]))
    print(new_idx)
    print(tokenizer.decode(new_idx))

if __name__ == '__main__':
    main()