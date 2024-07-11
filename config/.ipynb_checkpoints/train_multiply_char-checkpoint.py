# train a miniature character-level n-digit multiplication model
# good for debugging and playing on macbooks and such

out_dir = 'out-multiply-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'multiply-char'
wandb_run_name = 'mini-gpt'

dataset = 'multiply'
gradient_accumulation_steps = 1
batch_size = 64 # 64
block_size = 256 # 256. Context of up to 256 previous characters

#########  Model Scale #########

# # baby GPT model: gpt-2
# n_layer = 12
# n_head = 12
# n_embd = 768

# baby GPT model: mini-gpt
n_layer = 6
n_head = 6
n_embd = 384  # minGPT version: 192

# # baby GPT model: micro-gpt
# n_layer = 4
# n_head = 4
# n_embd = 128

# # baby GPT model: nano-gpt
# n_layer = 3
# n_head = 3
# n_embd = 48

##################################

dropout = 0.1

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 20000000
lr_decay_iters = 20000000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
decay_lr = False

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
