"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
from miditok import REMIPlus
from miditoolkit import MidiFile
import numpy as np
import torch
import tiktoken

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from model import GPTConfig, GPT
from encoder.encoder import MusicEncoder

# -----------------------------------------------------------------------------
tokenizer_config = 'data/maestro-v3.0.0/tokenizer_config.json'
out_dir = 'out-maestro-raw' # ignored if init_from is not 'resume'
num_samples = 10 # number of samples to draw
max_new_tokens = 25000 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('scripts/configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Load Tokenization
encoder = MusicEncoder(vocab_size=1534, load_config=True, tokenizer_config=tokenizer_config, use_bpe=True)
# encoder = MusicEncoder(vocab_size=50257, load_config=True, tokenizer_config=tokenizer_config, use_bpe=True)
x = torch.tensor(encoder.encode(["BOS_None"]).astype(np.int32), dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            filename = os.path.join(out_dir, f"{k}.midi")
            y = model.generate(x, max_new_tokens, temperature=1e-6, top_k=top_k)
            print(f"{encoder.decode(y[0].tolist()).tokens[:20]}\nSaving to {filename}")
            print('---------------')
            encoder.save_to_midi(y[0].tolist(), filename)

