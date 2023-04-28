out_dir = 'out-adl-raw'
eval_interval = 50
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'music-gpt'
wandb_run_name = 'adl-pop-raw'

dataset = 'adl-piano-midi/Pop'
vocab_size = 1534 # from encoder
batch_size = 4
gradient_accumulation_steps = 8
block_size = 512
# 33000000 training tokens / (1 batch size * 2^15 block size * 64) ~= 15.7 iterations per epoch
# 2.7 mil / 32 * 512 = 160 iters

# Use GPT2 parameters
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2
# layer = 6
# n_head = 6
# n_embd = 384
# dropout = 0.1

# learning_rate = 1e-4 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually
# min_lr = 1e-5 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
# warmup_iters = 100 # not super necessary potentially
learning_rate = 1e-5 # with baby networks can afford to go a bit higher
max_iters = 1000
lr_decay_iters = 1000 # make equal to max_iters usually
min_lr = 1e-7 # learning_rate / 10 usually
beta2 = 0.9 # make a bit bigger because number of tokens per iter is small
warmup_iters = 50 # not super necessary potentially  