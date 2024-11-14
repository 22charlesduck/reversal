import torch
from torch.utils.data import Dataset
import numpy as np
import random
import string
from torch.nn.utils.rnn import pad_sequence


def generate_and_save(n_nodes, n_test):
    """
    Generate a list of train and testing graphs with fixed-length numbers and save them for reproducibility
    """
    
    hash = set()
    # Open train file
    file = open('../data/datasets/reverse/' + 'train_normalfb' + str(n_nodes) + '.txt', 'w')
    testfile = open('../data/datasets/reverse/' + 'test_normalfb' + str(n_nodes) + '.txt', 'w')
    for i in range(n_nodes):
        stri = ''.join(random.choices(string.ascii_uppercase, k=10))
        while stri not in hash:
            stri = ''.join(random.choices(string.ascii_uppercase, k=10))
            hash.add(stri)
        mask = "A"*10
        out = f"=a{i}-b{i}>" 
        file.write('f' + out + '\n')
        # file.write(stri + out + '\n')
        if i %2 == 0:
            out = f"=b{i}-a{i}>"
            file.write('@' + out + '\n')
            # file.write(stri + out + '\n')
        else:
            out = f"=b{i}-a{i}>"
            testfile.write('@' + out + '\n')
    # for i in range(n_nodes):
    #     out = f"=a{i}-b{i}>" 
    #     file.write(out + '\n')

    # for i in range(n_nodes // 2):
    #     out = f"=b{i}-a{i}>"
    #     file.write(out + '\n')
    
    # for i in range(n_nodes):
    #     out = f"=a{i}-b{i}>" 
    #     file.write(out + '\n')

    # for i in range(n_nodes // 2):
    #     out = f"=b{i}-a{i}>"
    #     file.write(out + '\n')

    file.close()
    testfile.close()

    # # Open test file
    # file = open('../data/datasets/reverse/' + 'test_10maskfb' + str(n_nodes) + '.txt', 'w')
    # for i in range(n_nodes):
    #     if i %2 == 1:
    #         out = f"=b{i}-a{i}>"
    #         file.write(out + '\n')

    # file.close()

def prefix_target_list(filename=None, reverse=False, n_nodes =  2000):
    """
    Load graphs and split them into prefix and target and return the list
    """
    data_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        prefix = line.strip().split('=')[0] + '='
        target = line.strip().split('=')[1]
        if reverse:
            target = ','.join(target.split(',')[::-1])
        data_list.append((prefix, target))

    return data_list


class Reverse(Dataset):
    def __init__(self, tokenizer, n_samples, data_path, device, eval=False, teacherless_token=None, reverse=False, n_nodes = 5000):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.device = device
        self.eval_mode = eval
        self.data_path = data_path
        self.teacherless_token = teacherless_token
        self.reverse = reverse
        self.n_nodes = n_nodes

        self.data_file = prefix_target_list(self.data_path, reverse=reverse)[:n_samples]
        print(self.data_file)
        self.tokenized, self.num_prefix_tokens, self.num_target_tokens, self.num_eval_prefix, self.num_eval_target, self.max_target_len = tokenizer.tokenize(self.data_file)

        self.num_tokens = self.num_prefix_tokens + self.max_target_len

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        if self.eval_mode:
            # In eval mode return the entire sequence
            return self.tokenized[idx].to(self.device)

        # Create inputs
        x = self.tokenized[idx][:-1].clone()
        y = self.tokenized[idx][1:].clone()
        if idx % 2 == 1:
            y[:self.num_prefix_tokens - 1] = -1
        if self.teacherless_token is not None and idx %2 == 0 and not self.eval_mode:
            x[self.num_prefix_tokens:] = self.teacherless_token
        # Create targets in the form [-1, ..., -1, 4, 7, 9, 2, ...] where we replace the prefix tokens by -1 so that
        # we can skip their gradient calculation in the loss (double-check if that's correct)
        # y = torch.cat([-torch.ones((self.num_prefix_tokens - 1, )),
        #                self.tokenized[idx][self.num_prefix_tokens:].clone()])

        return x.to(self.device), y.long().to(self.device)

    def eval(self):
        # Switch to "eval" mode when generating sequences without teacher-forcing
        self.eval_mode = True

    def train(self):
        # Switch back to "train" mode for teacher-forcing
        self.eval_mode = False
    
    def collate_fn(self, batch):
        # Separate sequences and targets from batch
        sequences = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Pad sequences and targets to the length of the longest sequence in the batch with -1 padding
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=-1)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)

        # Create a mask for padded tokens (1 for real tokens, 0 for padding)
        attention_mask = (padded_sequences != -1).long()
        
        return padded_sequences, padded_targets, attention_mask


if __name__ == '__main__':
    import os
    print( os.getcwd())
    # Create graphs and save
    n_test = 5000
    generate_and_save(n_nodes = 20000, n_test=n_test)
