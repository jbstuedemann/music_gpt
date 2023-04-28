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
tokenizer_config = 'data/adl-piano-midi/tokenizer_config.json'
midi_file = 'context.midi'
# -----------------------------------------------------------------------------


# Load Tokenization
encoder = MusicEncoder(vocab_size=1534, load_config=True, tokenizer_config=tokenizer_config, use_bpe=False)

file = MidiFile(midi_file)
print(encoder.tokenizer(file).tokens)
