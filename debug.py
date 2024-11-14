#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from contextlib import nullcontext
import torch
from tqdm import tqdm

from data import get_dataset
from utils.training_utils import get_lr, get_run_name, AverageMeter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from evaluate import evaluate, evaluate_forced
from models import get_model
from tokenizing import get_tokenizer, NumeralTokenizer, Tokenizer
from data.reverse import prefix_target_list
import os
import wandb
from argparse import Namespace
import re

# In[2]:


get_lr(3*500, 1e-5, 125, 1000*3, 1e-6)


# In[2]:


args = {
    "model": 'gpt',
    "dataset": 'reverse',
    "n_train": 300,
    "n_test": 5000,
    "n_nodes": 100,
    "num_nodes": 20000,
    "deg": 2,
    "path_len": 5,
    "mate_in": 2,
    "unrolled": True,
    "batch_size": 128,
    "lr": 5e-4,
    "weight_decay": 1e-2,
    "epochs": 300,
    "save_every": 30,
    "teacherless": False,
    "reverse": False,
    "eval_train": False,
    "eval_every": 3750,
    "use_wandb": False,
    "wandb_entity": '5000',
    "n_layer": 36,
    "n_head": 20,
    "n_embd": 1280,
    "block_size": 11,
    "teacherless": True
}

args = Namespace(**args)


# In[31]:




# In[4]:


# System stuff
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"  # Only allow GPU 0, or "0,1" for multiple GPUs

# Clear the cached device count to ensure PyTorch re-evaluates available devices
torch.cuda.device_count.cache_clear()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device:', device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# Model stuff
top_k = 1

# Evaluation stuff
eval_iters = 1000
eval_interval = 5
log_interval = 10

# Optimiser
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
beta1 = 0.9
beta2 = 0.999
decay_lr = True
args.compile = False if device == 'cuda' else False
args.use_flash = True if device == 'cuda' else False
warmup_iters = 100
min_lr = 1e-6

run_name = get_run_name(args)
path = './checkpoints/' + run_name + '.pt'

# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)

checkpoint_dir = "../../../../data/user_data/clding/checkpoints_normalmaskfb10_3000"
checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint_epoch_1440.pt")

train_data, test_data = get_dataset(args, tokenizer, device)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn = train_data.collate_fn)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn = test_data.collate_fn)
target_len = train_data.num_tokens - train_data.num_prefix_tokens
max_iters = len(train_data) * args.epochs

lr_decay_iters = max_iters

block_size = train_data.num_tokens
args.block_size = 17
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None

# Load model and move to device
model = get_model(args)
print("checkpoint_path", checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location=device)
new_state_dict = {}
for k, v in checkpoint.items():
    new_key = k.replace("module.", "")  # Remove 'module.' prefix
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()


# Load a single data point from train_data
# Assuming train_data is directly accessible and each item is formatted correctly
data_path = './data/datasets/reverse/'
train_path, test_path = data_path + f'train_{args.n_nodes}.txt', data_path + f'test_{args.n_nodes}.txt'


# In[30]:


tokenizer = get_tokenizer(args)
print(tokenizer.encode('=a2'))


# In[40]:




# In[44]:





# In[48]:


# data_list = prefix_target_list(train_path, reverse=args.reverse)
# data_point = data_list[300][0] + data_list[300][1] # Adjust based on train_data's structure
# data_point = data_point.split("-")[0]


# In[47]:


# data_list = prefix_target_list(train_path, reverse=args.reverse)
# data_point = data_list[300][0] + data_list[300][1] # Adjust based on train_data's structure
# data_point = data_point.split("-")[0]
count = 0
testfile = './data/datasets/reverse/' + 'test_normal_10hashfb20000' + '.txt'

pattern = re.compile(r"^(.+?=b\d+)")

# List to store prefixes
prefixes = []

# Open the file and process each line
with open(testfile, "r") as file:
    for line in file:
        line1 = line.split('@=')
        p
        line = line1[1].split('-')
        i = int(line[0].split('b')[1])
        if i >= 4000:
            break
        input_tokens = tokenizer.encode(+'@=' + line[0])
        # if isinstance(data_point, str):
        #     # Tokenize directly if raw text
        #     input_tokens = tokenizer.encode(data_point)
        # else:
        #     # If data_point is structured, assume "input_text" is the correct key
        #     input_tokens = tokenizer.encode(data_point["input_text"])

        # print("input_tokens", input_tokens)
        input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)  # Add batch dimension

        # Initialize the generated tokens with the input prompt
        generated_tokens = input_tokens.copy()

        # Set maximum generation length to prevent infinite loops
        max_length = 7  # You can adjust this as needed
        # print("input_tokens", input_tokens)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                for _ in range(max_length):
                    # Convert the current generated tokens to tensor for each step
                    input_tensor = torch.tensor(generated_tokens).unsqueeze(0).to(device)
                    attn_mask = (input_tensor != -2).long()  # Adjust padding if needed
                    # print("input_tensor", input_tensor)
                    
                    # Forward pass through the model
                    logits, _, _ = model(input_tensor)
                    
                    # Get the prediction for the next token (only the last position)
                    
                    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
                    # Append the predicted token to the generated sequence
                    generated_tokens.append(next_token_id)
                    if tokenizer.decode([next_token_id])[0] == ">":
                        break

        # Decode the complete generated token sequence into text
        prediction_text = tokenizer.decode(generated_tokens)
        prediction_text = "".join(prediction_text)
        print("Generated Text:", prediction_text)
        if (prediction_text == line[0]+f"-a{i}>"):
            count+=1
        print(count)
print(count)



# In[ ]:




