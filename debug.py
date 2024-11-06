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
from tokenizing import get_tokenizer
from data.reverse import prefix_target_list
import os
import wandb

# Parse arguments
parser = argparse.ArgumentParser(description="Next-token failures")
# Data
parser.add_argument(
    "--model", default='gpt2', type=str, help="Type of model"
    )
parser.add_argument(
    "--dataset", default='reverse', type=str, help="Choice of dataset"
    )
parser.add_argument(
    "--n_train", default=200000, type=int, help="Number of training samples"
    )
parser.add_argument(
    "--n_test", default=5000, type=int, help="Number of test samples"
    )
parser.add_argument(
    "--n_nodes", default=50, type=int, help="Number of nodes for reversal"
)
parser.add_argument(
    "--num_nodes", default=50, type=int, help="Number of node values in graph"
    )
parser.add_argument(
    "--deg", default=2, type=int, help="Degree of starting node"
    )
parser.add_argument(
    "--path_len", default=5, type=int, help="Path length in star graph"
    )
parser.add_argument(
        "--mate_in", default=2, type=int, help="For chess, number of moves to checkmate"
    )
parser.add_argument(
        "--unrolled", action=argparse.BooleanOptionalAction, default=True, help="For chess, unrolled board state",
    )
parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size",
    )
parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate",
    )
parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Strength of weight decay",
    )
parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs",
    )
parser.add_argument(
        "--save_every", type=int, default=60, help="Interval (in steps) at which to save model",
    )
parser.add_argument(
        "--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training",
    )
parser.add_argument(
        "--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets",
    )
parser.add_argument(
        "--eval_train", action=argparse.BooleanOptionalAction, default=False, help="Eval for training set",
    )
parser.add_argument(
        "--eval_every", type=int, default=3750, help="Interval (in steps) to evaluate the model on test",
    )
parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Whether to use wandb",
    )
parser.add_argument(
        "--wandb_entity", type=str, default=5000, help="Wandb username",
    )


args = parser.parse_args()
# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
wandb_entity = args.wandb_entity
wandb_log = args.use_wandb
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

checkpoint_dir = "../../../../data/user_data/clding/checkpoints"
checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint_epoch_40.pt")

train_data, test_data = get_dataset(args, tokenizer, device)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn = train_data.collate_fn)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn = test_data.collate_fn)
target_len = train_data.num_tokens - train_data.num_prefix_tokens
max_iters = len(train_data) * args.epochs

lr_decay_iters = max_iters

block_size = train_data.num_tokens
args.block_size = train_data.num_tokens
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
data_list = prefix_target_list(train_path, reverse=args.reverse)
data_point = data_list[300][0] + data_list[300][1] # Adjust based on train_data's structure
data_point = data_point.split("-")[0]
data_point = '=a00004'
print(data_point)
if isinstance(data_point, str):
    # Tokenize directly if raw text
    input_tokens = tokenizer.encode(data_point)
else:
    # If data_point is structured, assume "input_text" is the correct key
    input_tokens = tokenizer.encode(data_point["input_text"])

print("input_tokens", input_tokens)
input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)  # Add batch dimension

# Initialize the generated tokens with the input prompt
generated_tokens = input_tokens.copy()

# Set maximum generation length to prevent infinite loops
max_length = 7  # You can adjust this as needed
print("input_tokens", input_tokens)
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for _ in range(max_length):
            # Convert the current generated tokens to tensor for each step
            input_tensor = torch.tensor(generated_tokens).unsqueeze(0).to(device)
            attn_mask = (input_tensor != -2).long()  # Adjust padding if needed
            
            # Forward pass through the model
            logits, _, _ = model(input_tensor, attn_mask=attn_mask)
            
            # Get the prediction for the next token (only the last position)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            print(tokenizer.decode([next_token_id]))
            
            # Append the predicted token to the generated sequence
            generated_tokens.append(next_token_id)

# Decode the complete generated token sequence into text
prediction_text = tokenizer.decode(generated_tokens)
print("Generated Text:", prediction_text)

print("Prediction Text:", prediction_text)
