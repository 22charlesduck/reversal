import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
import wandb

from tokenizing import get_tokenizer
from utils.training_utils import get_lr, get_run_name, AverageMeter
from data import get_dataset
from evaluate import evaluate, evaluate_forced
from models import get_model

# Parse arguments
parser = argparse.ArgumentParser(description="Next-token prediction")
# Data
parser.add_argument("--model", type=str, default='gpt', help="Learning rate")
parser.add_argument("--dataset", default='graph', type=str, help="Choice of dataset")
parser.add_argument("--n_train", default=200000, type=int, help="Number of training samples")
parser.add_argument("--n_test", default=10000, type=int, help="Number of test samples")
parser.add_argument("--num_nodes", default=20000, type=int, help="Number of node values in graph")
parser.add_argument("--n_nodes", default=20000, type=int, help="Number of node values in graph")
parser.add_argument("--deg", default=2, type=int, help="Degree of starting node")
parser.add_argument("--path_len", default=5, type=int, help="Path length in star graph")
parser.add_argument("--mate_in", default=2, type=int, help="For chess, number of moves to checkmate")
parser.add_argument("--unrolled", action=argparse.BooleanOptionalAction, default=True, help="For chess, unrolled board state")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Strength of weight decay")
parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
parser.add_argument("--save_every", type=int, default=120, help="Interval (in steps) at which to save model")
parser.add_argument("--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training")
parser.add_argument("--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets")
parser.add_argument("--eval_train", action=argparse.BooleanOptionalAction, default=False, help="Eval for training set")
parser.add_argument("--eval_every", type=int, default=3750, help="Interval (in steps) to evaluate the model on test")
parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Whether to use wandb")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb username")
parser.add_argument("--n_layer", type=int, default=36, help="Number of layers for the transformer model")
parser.add_argument("--n_head", type=int, default=20, help="Number of attention heads")
parser.add_argument("--n_embd", type=int, default=1280, help="Embedding dimension size")
parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="Resume training from checkpoint")

args = parser.parse_args()

import os
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"  # Only allow GPU 0, or "0,1" for multiple GPUs

# Clear the cached device count to ensure PyTorch re-evaluates available devices
torch.cuda.device_count.cache_clear()

# System stuff
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
wandb_entity = args.wandb_entity
wandb_log = args.use_wandb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Model setup
start_epoch = 0
top_k = 1
eval_iters = 1000
eval_interval = 5
log_interval = 10
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
beta1 = 0.9
beta2 = 0.999
decay_lr = True
args.compile = False if device == 'cpu' else False
args.use_flash = True if device[:4] == 'cuda' else False
warmup_iters = 125
min_lr = 1e-6

run_name = get_run_name(args)
path = './checkpoints/' + run_name + '.pt'

# Get tokenizer and data
tokenizer = get_tokenizer(args)
train_data, test_data = get_dataset(args, tokenizer, device)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=test_data.collate_fn)

target_len = train_data.num_tokens - train_data.num_prefix_tokens
max_iters = args.n_train * args.epochs//args.batch_size
lr_decay_iters = max_iters
args.block_size = 17
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None

# Model initialization
model = get_model(args)
if args.compile:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)
if args.resume:
    checkpoint_dir = "../../../../data/user_data/clding/checkpoints_maskfb10_12000"
    checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint_epoch_840.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
print(device)
try:
    model.to(device)
    print("Model moved to", device)
except Exception as e:
    print(f"Failed to move model to {device}: {e}")
    torch.cuda.empty_cache()  # Clear GPU memory if any allocation happened
    device = torch.device("cpu")
    model.to(device)
    print("Model moved to CPU as fallback")
# model.to(device)
# model = torch.nn.DataParallel(model)
# model.to(device)
model.train()

# Optimizer and context setup
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# Wandb logging setup
if wandb_log:
    wandb.init(project='next-token-failures', entity=wandb_entity, config=args.__dict__)
    wandb.run.name = run_name

results = {}
num_iters = 0
if args.resume:
    start_epoch = 841
    num_iters = start_epoch * args.n_train // args.batch_size

for ep in range(start_epoch, args.epochs):
    train_bar = tqdm(train_loader)
    total_loss, total_acc = AverageMeter(), AverageMeter()

    for x, y, attn_mask in train_bar:
        # print(x.shape, y.shape, attn_mask.shape)
        x, y, attn_mask = x.to(device), y.to(device), attn_mask.to(device)
        # print('hi')
        if ep % args.save_every == 0 and ep > 500:
            checkpoint_dir = f"../../../../data/user_data/clding/checkpoints_normalfb_{args.n_train}"
            checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint_epoch_{ep}.pt")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        # print('hi1')
        # Learning rate scheduling
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print('hi2')
        with ctx:
            # print("Data prepared:", x.shape, y.shape, attn_mask.shape)
            logits, loss, accs = model(x, y, attn_mask)
        
        loss = loss.mean()
        accs['acc'] = accs['acc'].mean().item()
        total_loss.update(loss.item(), x.shape[0] * train_data.num_target_tokens)
        total_acc.update(accs['acc'], x.shape[0] * train_data.num_target_tokens)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        num_iters += 1

        train_bar.set_description(
            'Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'.format(ep, args.epochs, total_loss.get(), total_acc.get(percentage=True))
        )

        # Evaluation
        if num_iters % args.eval_every == 0 and num_iters > 1:
            try:
                if args.eval_train:
                    results = evaluate(model, train_loader, temperature=0.8, top_k=top_k, results=results, mode='train')
                    results = evaluate_forced(model, train_loader, results=results, mode='train')

                results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='test')
                results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='test')

                if wandb_log:
                    wandb.log(results)
            except Exception as e:
                print(e)
                continue

    if wandb_log:
        wandb.log({
            "train/loss": total_loss.get(),
            "train/accuracy": total_acc.get(percentage=True),
            "learning_rate": lr,
            "iteration": num_iters
        })
    
    print(lr)
