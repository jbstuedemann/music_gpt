out_dir = 'out-maestro-wrapper'
eval_interval = 25
eval_iters = 200
log_interval = 1

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'music-gpt'
wandb_run_name = 'wrapper_model-maestro'

dataset = 'maestro-v3.0.0'
vocab_size = 1534 # from encoder
batch_size = 1
gradient_accumulation_steps = 32
# 33000000 training tokens / (1 batch size * 2^15 block size * 64) ~= 15.7 iterations per epoch
# 2.7 mil / 32 * 512 = 160 iters

learning_rate = 1e-6 # with baby networks can afford to go a bit higher
max_iters = 200
lr_decay_iters = 200 # make equal to max_iters usually
min_lr = 1e-8# learning_rate / 10 usually
beta2 = 0.9 # make a bit bigger because number of tokens per iter is small
warmup_iters = 50 # not super necessary potentially  