import tiktoken
import torch
from typing import Tuple

'''-------------------------------------------------------------------------------------'''
# A dataloader to fetch data for training and testing

class BatchLoader:
    def __init__(self, batch_size: int = 8, context_length: int = 4) -> None:
        self.batch_size = batch_size
        self.context_length = context_length # maximum number of tokens at a single time
        with open('input.txt','r') as file: content = file.read() #reading the text contents of the file
        self.tokenizer = tiktoken.encoding_for_model("gpt-2") #set the tokenizer to gpt-2 version
        self.encoded_content = self.tokenizer.encode(content) #tokenize/encode the entire input dataset
        self.curr_idx = 0
        
    def _get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.tensor(self.encoded_content[self.curr_idx:self.curr_idx + self.batch_size*self.context_length])
        Y = torch.tensor(self.encoded_content[self.curr_idx+1:self.curr_idx + self.batch_size*self.context_length+1])
        return X.view(self.batch_size,self.context_length), Y.view(self.batch_size,self.context_length)
    
    def next_batch(self):
        if self.curr_idx + self.batch_size*self.context_length +1 <= len(self.encoded_content):
            X, Y = self._get_batch()
            self.curr_idx += self.batch_size*self.context_length
        else:
            self.curr_idx = 0
            X, Y = self._get_batch()
            self.curr_idx += self.batch_size*self.context_length
        return X, Y
            
if __name__ == '__main__':
    loaders = BatchLoader()
    x1, y1 = loaders.next_batch()
    x2, y2 = loaders.next_batch()
    print(x1,y1)