import torch
from transformers import AutoTokenizer
from tokenizing.numeral_tokenizer import NumeralTokenizer
import string


class Tokenizer:
    def __init__(self, encoder, decoder, vocab_size, name=None):
        self.encode = encoder
        self.decode = decoder
        self.vocab_size = vocab_size
        self.name = name

    def tokenize(self, data_list):
        """
        Takes a list of prefix-target pairs, tokenizes and concatenates them
        """
        out = []
        str = data_list[0][0] + data_list[0][1]
        p = str.split('-')[0]+'-'
        t = str.split('-')[1]
        max_target_len = len(self.encode(data_list[0][1]))
        eval_prefix = len(self.encode(p))
        eval_target = len(self.encode(t))
        prefix_len = len(self.encode(data_list[1][0]))
        target_len = len(self.encode(data_list[1][1]))
        same_len = True

        for prefix, target in data_list:
            orig_prefix = prefix
            orig_target = target
            prefix = torch.tensor(self.encode(prefix))
            target = torch.tensor(self.encode(target))
            if not (len(prefix) == prefix_len and len(target) == target_len):
                same_len = False
                max_target_len = max(max_target_len, len(target))
            seq = torch.concatenate([prefix, target], dim=-1).long()
            out.append(seq)

        # Check if all prefixes and all targets have the same length
        if not same_len:
            print('Not all prefixes or targets have the same length!!')
        else:
            print('Equal sequence lengths!')
        
        return out, prefix_len, target_len, eval_prefix, eval_target, max_target_len


def get_tokenizer(args):
    if args.model == 'gpt':
        t = NumeralTokenizer(args.num_nodes)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=2*args.num_nodes + 6 + len(string.ascii_uppercase), name='numeral')
    elif args.model.startswith('gpt2'):
        # print('hi1')
        t = AutoTokenizer.from_pretrained('gpt2')
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=50257 , name='gpt2')
    elif args.model.startswith('pythia'):
        # print('hi1')
        t = AutoTokenizer.from_pretrained('EleutherAI/' + args.model)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=50304, name='gpt2')
    elif args.model.startswith('phi'):
        # print('hi1')
        t = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=51200, name='phi')

    return tokenizer


# args = {
#     "model": 'gpt',
#     "dataset": 'reverse',
#     "n_train": 30000,
#     "n_test": 5000,
#     "n_nodes": 20000,
#     "num_nodes": 20000,
#     "deg": 2,
#     "path_len": 5,
#     "mate_in": 2,
#     "unrolled": True,
#     "batch_size": 128,
#     "lr": 5e-4,
#     "weight_decay": 1e-2,
#     "epochs": 300,
#     "save_every": 60,
#     "teacherless": False,
#     "reverse": False,
#     "eval_train": False,
#     "eval_every": 3750,
#     "use_wandb": False,
#     "wandb_entity": '5000',
#     "n_layer": 36,
#     "n_head": 20,
#     "n_embd": 1280,
#     "block_size": 6
# }

# args = Namespace(**args)
# get_tokenizer(args)