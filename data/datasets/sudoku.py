import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import random

class Sudoku(Dataset):
    def __init__(self, tokenizer, n_samples, device, eval=False, teacherless_token=None, reverse=False):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.device = device
        self.eval_mode = eval
        self.teacherless_token = teacherless_token
        self.reverse = reverse

        if eval:
            base_data = load_dataset('Ritvik19/Sudoku-Dataset', split='test')
        else:
            base_data = load_dataset('Ritvik19/Sudoku-Dataset', split='train')

        self.data_file = base_data['puzzle'][:n_samples]
        self.tokenized, self.num_prefix_tokens, self.num_target_tokens = tokenizer.tokenize(self.data_file)

        self.num_tokens = self.num_prefix_tokens + self.num_target_tokens

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        if self.eval_mode:
            # In eval mode return the entire sequence
            return self.tokenized[idx].to(self.device)

        # Create inputs
        x = self.tokenized[idx][:-1].clone()
        if self.teacherless_token is not None:
            x[self.num_prefix_tokens:] = self.teacherless_token
            x = x.to(self.device)
        # Create targets in the form [-1, ..., -1, 4, 7, 9, 2, ...] where we replace the prefix tokens by -1 so that
        # we can skip their gradient calculation in the loss (double-check if that's correct)
        y = torch.cat([-torch.ones((self.num_prefix_tokens - 1, )),
                       self.tokenized[idx][self.num_prefix_tokens:].clone()])

        return x.to(self.device), y.long().to(self.device)

    def eval(self):
        # Switch to "eval" mode when generating sequences without teacher-forcing
        self.eval_mode = True

    def train(self):
        # Switch back to "train" mode for teacher-forcing
        self.eval_mode = False


467100805
912835607
085647192
296351470
708920351
531408926
073064510
624519783
159783064