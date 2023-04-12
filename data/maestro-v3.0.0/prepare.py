from miditok import REMIPlus
from miditoolkit import MidiFile
import numpy as np
import os
import pandas as pd
from path import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from encoder.encoder import MusicEncoder

VOCAB_SIZE = 1534

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "maestro-v3.0.0.csv"))
encoder = MusicEncoder(VOCAB_SIZE, load_config=False, tokenizer_config=os.path.join(os.path.dirname(__file__), "tokenizer_config.json"))
encoder.train_bpe([os.path.join(os.path.dirname(__file__), filename) for filename in df['midi_filename']])
print(f"Trained BPE with vocab size of {len(encoder.tokenizer)}")

for split in ('train', 'validation'):
    
    #Tokenize Data
    split_df = df.loc[df['split'] == split]
    duration = split_df['duration'].sum()
    print(f"({split}) Found {len(split_df)} MIDI files for total length of {'%d:%02d:%02d'%(duration//3600,(duration//60)%60,duration%60)}")

    dataset = encoder.tokenize_dataset([os.path.join(os.path.dirname(__file__), filename) for filename in split_df['midi_filename']])

    print(f"{split} has {len(dataset):,} tokens")
    dataset.tofile(os.path.join(os.path.dirname(__file__), f"{split if split == 'train' else 'val'}.bin"))