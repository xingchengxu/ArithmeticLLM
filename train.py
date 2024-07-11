"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import random
import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from torch.utils.tensorboard import SummaryWriter

from eval_arithmetic import get_eval_data, estimate_accuracy, estimate_accuracy_digit_by_digit, collect_data_zeta_chi, transform_dig_dummy_data, get_dig_dummy_r2
from sklearn.tree import DecisionTreeRegressor


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
ckpt_name = 'ckpt.pt'
seed = 1337
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# tensorboard logging
tensorboard_log = False
run_id = 0
digit_wise_eval = False
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
line_train = False
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
slow_attention = False
n_eval = 250
save_by_stage = False
use_embedding = None
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
if line_train:  # for math euqation, line-by-line training
    with open(os.path.join(data_dir, 'train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, 'val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
else:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
# best_ood_modval_acc = 0.0

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    stoi, itos = meta['stoi'], meta['itos']

# Data loader function for math euqation, line-by-line training
def get_batch_line_train(split, stoi):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data), batch_size)
    x = [torch.tensor(data[i][:-1], dtype=torch.long) for i in ix]
    y = [torch.tensor(data[i][1:], dtype=torch.long) for i in ix]
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=stoi['<eos>'])
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=stoi['<eos>'])

    # # Create a new tensor with the desired block size, filled with eos_value to a shape of batch_size X block_size
    # x = torch.full((x_immed.shape[0], block_size), stoi['<eos>'], dtype=x_immed.dtype)
    # x[:, :x_immed.shape[1]] = x_immed

    # y = torch.full((y_immed.shape[0], block_size), stoi['<eos>'], dtype=y_immed.dtype)
    # y[:, :y_immed.shape[1]] = y_immed

    # we will only train in the output locations. -1 will mask loss to zero
    # Find the position of the equals sign in each sequence and mask positions before it
    for i in range(batch_size):
        eq_pos = (y[i] == stoi['=']).nonzero(as_tuple=True)[0]
        if eq_pos.size(0) > 0:  # If an equals sign is found
            y[i, :eq_pos[0]+1] = -1  # Mask positions before and including the equals sign

    # # Find the position of the <eos> sign in each sequence and mask positions after it
    # for i in range(batch_size):
    #     eos_pos = (y[i] == stoi['<eos>']).nonzero(as_tuple=True)[0]
    #     if eos_pos.size(0) > 0:  # If an <eos> sign is found
    #         y[i, eos_pos[0]:] = -1  # Mask positions after and including the <eos> sign
    
    if torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, slow_attention, use_embedding)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, slow_attention, use_embedding)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith(('gpt2','./GPT2/gpt2')):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

print(f"Using device:{device}")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if line_train:
                X, Y = get_batch_line_train(split, stoi)
            else:
                X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def estimate_accuracy_digit():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        correct = 0
        total = 0
        for k in range(eval_iters):
            if line_train:
                X, Y = get_batch_line_train(split, stoi)
            else:
                X, Y = get_batch(split)
            with ctx:
                logits, _ = model(X, Y)
            predictions = torch.argmax(logits, dim=-1)
            
            # Create a mask to ignore elements with -1 in Y
            mask = (Y != -1)
            valid_predictions = predictions[mask]
            valid_Y = Y[mask]
            
            correct += (valid_predictions == valid_Y).sum().item()
            total += valid_Y.numel()
        out[split] = correct / total if total > 0 else 0
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# wandb logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# tensorboard logging
if tensorboard_log and master_process:
    run_name = f"addition_run{run_id}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

# training loop
if line_train:
    X, Y = get_batch_line_train('train', stoi)
else:
    X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0


########################################
############ load Eval Data ############
########################################
eval_train = get_eval_data(data_dir, data_name='input.txt', split='train', n_eval=128)
eval_test = get_eval_data(data_dir, data_name='input.txt', split='test', n_eval=128)

if digit_wise_eval:
    # # for Standard Addtion
    # eval_data = {}
    # data1_name = 'addition_dataset_1_reversed_prepend.txt'  # addition_dataset_3_reversed, addition_dataset_7_100k_reversed
    # eval_data1 = get_eval_data(data_dir, data_name=data1_name, split='eval', n_eval=n_eval)
    # eval_data[1] = eval_data1
    # data2_name = 'addition_dataset_ab2_100k_reversed_1.txt'  # addition_dataset_3_reversed, addition_dataset_7_100k_reversed
    # eval_data2 = get_eval_data(data_dir, data_name=data2_name, split='eval', n_eval=n_eval)
    # eval_data[2] = eval_data2
    # data3_name = 'addition_dataset_ab3_100k_reversed_1.txt'  # addition_dataset_3_reversed, addition_dataset_7_100k_reversed
    # eval_data3 = get_eval_data(data_dir, data_name=data3_name, split='eval', n_eval=n_eval)
    # eval_data[3] = eval_data3
    # data4_name = 'addition_dataset_ab4_100k_reversed_1.txt'  # addition_dataset_3_reversed, addition_dataset_7_100k_reversed
    # eval_data4 = get_eval_data(data_dir, data_name=data4_name, split='eval', n_eval=n_eval)
    # eval_data[4] = eval_data4
    # data5_name = 'addition_dataset_ab5_100k_reversed_1.txt'  # addition_dataset_3_reversed, addition_dataset_7_100k_reversed
    # eval_data5 = get_eval_data(data_dir, data_name=data5_name, split='eval', n_eval=n_eval)
    # eval_data[5] = eval_data5
    # data6_name = 'addition_dataset_ab6_100k_reversed_1.txt'  # addition_dataset_3_reversed, addition_dataset_7_100k_reversed
    # eval_data6 = get_eval_data(data_dir, data_name=data6_name, split='eval', n_eval=n_eval)
    # eval_data[6] = eval_data6
    # data7_name = 'addition_dataset_ab7_100k_reversed_1.txt'  # addition_dataset_3_reversed, addition_dataset_7_100k_reversed
    # eval_data7 = get_eval_data(data_dir, data_name=data7_name, split='eval', n_eval=n_eval)
    # eval_data[7] = eval_data7
    # data8_name = 'addition_dataset_ab8_100k_reversed_1.txt'  # addition_dataset_3_reversed, addition_dataset_7_100k_reversed
    # eval_data8 = get_eval_data(data_dir, data_name=data8_name, split='eval', n_eval=n_eval)
    # eval_data[8] = eval_data8
    # data9_name = 'addition_dataset_9_100k_reversed_prepend.txt'  # addition_dataset_3_reversed, addition_dataset_7_100k_reversed
    # eval_data9 = get_eval_data(data_dir, data_name=data9_name, split='eval', n_eval=n_eval)
    # eval_data[9] = eval_data9

    # for Modular Addtion with modulus p
    modulus_p = 101
    eval_data = {}
    data1_name = f'mod_addition_p{modulus_p}_ab1_100k_reversed_1.txt'
    eval_data1 = get_eval_data(data_dir, data_name=data1_name, split='eval', n_eval=n_eval)
    eval_data[1] = eval_data1
    data2_name = f'mod_addition_p{modulus_p}_ab2_100k_reversed_1.txt'
    eval_data2 = get_eval_data(data_dir, data_name=data2_name, split='eval', n_eval=n_eval)
    eval_data[2] = eval_data2
    data3_name = f'mod_addition_p{modulus_p}_ab3_100k_reversed_1.txt'
    eval_data3 = get_eval_data(data_dir, data_name=data3_name, split='eval', n_eval=n_eval)
    eval_data[3] = eval_data3
    data4_name = f'mod_addition_p{modulus_p}_ab4_100k_reversed_1.txt'
    eval_data4 = get_eval_data(data_dir, data_name=data4_name, split='eval', n_eval=n_eval)
    eval_data[4] = eval_data4
    data5_name = f'mod_addition_p{modulus_p}_ab5_100k_reversed_1.txt'
    eval_data5 = get_eval_data(data_dir, data_name=data5_name, split='eval', n_eval=n_eval)
    eval_data[5] = eval_data5
    data6_name = f'mod_addition_p{modulus_p}_ab6_100k_reversed_1.txt'
    eval_data6 = get_eval_data(data_dir, data_name=data6_name, split='eval', n_eval=n_eval)
    eval_data[6] = eval_data6
    data7_name = f'mod_addition_p{modulus_p}_ab7_100k_reversed_1.txt'
    eval_data7 = get_eval_data(data_dir, data_name=data7_name, split='eval', n_eval=n_eval)
    eval_data[7] = eval_data7
    data8_name = f'mod_addition_p{modulus_p}_ab8_100k_reversed_1.txt'
    eval_data8 = get_eval_data(data_dir, data_name=data8_name, split='eval', n_eval=n_eval)
    eval_data[8] = eval_data8
    data9_name = f'mod_addition_p{modulus_p}_ab9_100k_reversed_1.txt'
    eval_data9 = get_eval_data(data_dir, data_name=data9_name, split='eval', n_eval=n_eval)
    eval_data[9] = eval_data9

    eval_data_agg = []
    for k, v in eval_data.items():
        eval_data_agg = eval_data_agg + v

##################################################
############ Finish Loading Eval Data ############
##################################################

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        accuracy_digit = estimate_accuracy_digit()

        model.eval()
        with torch.no_grad():
            train_accuracy = estimate_accuracy(model, eval_train, data_dir, device, max_new_tokens=23, batch_size=128)
            test_accuracy = estimate_accuracy(model, eval_test, data_dir, device, max_new_tokens=23, batch_size=128)
            if digit_wise_eval:
                acc_dict = {}
                r2_dict = {}
                for k, data in eval_data.items():
                    acc_dict[k] = estimate_accuracy(model, eval_data[k], data_dir, device, max_new_tokens=23, batch_size=128)
                _, dig_acc = estimate_accuracy_digit_by_digit(model, eval_data_agg, data_dir, batch_size=128, device=device, verbose=False, debug=False)
                digit_dummy_data_list = []
                R2_VALID = 1
    
                for i in range(1, 10):
                    if i == 1:
                        digit_dummy_data_tmp = collect_data_zeta_chi(model, eval_data[i], data_dir, batch_size=64, device=device, 
                                                                     prepend=True, verbose=False, debug=False, 
                                                                     random_impute=True, impute_side=random.randint(1, 3) ,impute_repeat_num=4, 
                                                                     impute_digit_list=[i+1,i+2,i+3], impute_number=random.randint(1, 3))
                    else:
                        digit_dummy_data_tmp = collect_data_zeta_chi(model, eval_data[i], data_dir, batch_size=64, device=device, 
                                                                     prepend=True, verbose=False, debug=False, 
                                                                     random_impute=True, impute_side=random.randint(1, 3) ,impute_repeat_num=4,
                                                                     impute_digit_list=[i-1,i+1,i+2,i+3],impute_number=random.randint(1, 4))
                    digit_dummy_data_list.append(digit_dummy_data_tmp)
                digit_dummy_data = pd.concat(digit_dummy_data_list).reset_index(drop=True)
                transformed_data = transform_dig_dummy_data(digit_dummy_data)
                decision_tree_regressor = DecisionTreeRegressor(random_state=42)
                try:
                    r2_dict = get_dig_dummy_r2(transformed_data, decision_tree_regressor, rep=1)
                    R2_VALID = 1
                except:
                    R2_VALID = 0

            # train_accuracy = estimate_accuracy(model, data_dir, device, data_name='input.txt', split='train', n_eval=10000, batch_size=128)
            # test_accuracy = estimate_accuracy(model, data_dir, device, data_name='input.txt', split='test', n_eval=10000, batch_size=128)
            # ood_test_accuracy = estimate_accuracy(model, data_dir, device,  data_name='addition_dataset_4_100k_reversed.txt', split='ood_test', n_eval=10000, batch_size=128)
            # ood_test_mod_accuracy = estimate_accuracy(model, data_dir, device,  data_name='addition_dataset_4_100k_reversed.txt', split='ood_test', n_eval=10000, batch_size=128, modular=True)
        model.train()
        
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f},",
        f"train accdgt {accuracy_digit['train']:.4f}, val accdgt {accuracy_digit['val']:.4f}",
        f"train acc {train_accuracy:.4f}, id test acc {test_accuracy:.4f}")  # , ood test acc {ood_test_accuracy:.4f}, ood test mod acc {ood_test_mod_accuracy:.4f}
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        
        if tensorboard_log:
            model.eval()
            with torch.no_grad():
                writer.add_scalar(f"train_loss/train_loss", torch.Tensor(losses['train']), iter_num)
                writer.add_scalar(f"id_test_loss/id_test_loss", torch.Tensor(losses['val']), iter_num)
                writer.add_scalar(f"train_acc_dgt/train_acc_dgt", torch.Tensor([accuracy_digit['train']]), iter_num)
                writer.add_scalar(f"id_test_acc_dgt/id_test_acc_dgt", torch.Tensor([accuracy_digit['val']]), iter_num)
                writer.add_scalar(f"train_acc/train_acc", torch.Tensor([train_accuracy]), iter_num)
                writer.add_scalar(f"id_test_acc/id_test_acc", torch.Tensor([test_accuracy]), iter_num)
                # writer.add_scalar(f"ood_test_acc/ood_test_acc", torch.Tensor([ood_test_accuracy]), iter_num)
                # writer.add_scalar(f"ood_test_modacc/ood_test_modacc", torch.Tensor([ood_test_mod_accuracy]), iter_num)
                writer.add_scalar(f"learning_rate", torch.Tensor([lr]), iter_num)
                writer.add_scalar(f"model_flops_utilization", torch.Tensor([running_mfu*100]), iter_num)
                if digit_wise_eval:
                    for index, acc in acc_dict.items():
                        writer.add_scalar(f"eval_acc/acc{index}", torch.Tensor([acc]), iter_num)
                    for index, acc in dig_acc.items():
                        writer.add_scalar(f"digit_acc/acc{index}", torch.Tensor([acc]), iter_num)
                    if R2_VALID:
                        for index, r2 in r2_dict.items():
                            writer.add_scalar(f"digit_r2/r2_{index}_in_sample", torch.Tensor([r2['R2_in_sample']]), iter_num)
                            writer.add_scalar(f"digit_r2/r2_{index}_out_sample", torch.Tensor([r2['R2_out_of_sample']]), iter_num)
            model.train()

        # Check if we need to update the best validation loss
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            should_save_checkpoint = True
        else:
            should_save_checkpoint = False
    
        # # Check if we need to update the best OOD modified validation loss
        # if ood_test_mod_accuracy > best_ood_modval_acc:
        #     best_ood_modval_acc = ood_test_mod_accuracy
        #     should_save_checkpoint = True

        if should_save_checkpoint or always_save_checkpoint:
            # best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
    
    if iter_num == 0 and eval_only:
        break

    if save_by_stage:
        if iter_num <= 200000:
            if iter_num % 10000 == 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir} at iter_num {iter_num}")
                torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_'+ckpt_name))
        
        elif iter_num > 200000 and iter_num <= 1000000:
            if iter_num % 100000 == 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir} at iter_num {iter_num}")
                torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_'+ckpt_name))
        else:
            if iter_num % 200000 == 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir} at iter_num {iter_num}")
                torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_'+ckpt_name))
        
    if iter_num == int(max_iters/2):
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        print(f"saving checkpoint to {out_dir} at iter_num {iter_num}")
        torch.save(checkpoint, os.path.join(out_dir, 'half_'+ckpt_name))

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        if line_train:
            X, Y = get_batch_line_train('train', stoi)
        else:
            X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if tensorboard_log:
    writer.close()

checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
print(f"saving checkpoint to {out_dir} at iter_num {iter_num}")
torch.save(checkpoint, os.path.join(out_dir, 'final_'+ckpt_name))

if ddp:
    destroy_process_group()
