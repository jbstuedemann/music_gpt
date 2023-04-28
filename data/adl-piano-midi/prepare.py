# https://github.com/lucasnfe/adl-piano-midi

import miditoolkit
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from encoder.encoder import MusicEncoder


# Create dataset from one genre
genre = "Pop"
seed = 1234
val_percent = 0.1
VOCAB_SIZE = 1534


# Find all MIDI files
root = os.path.join(os.path.dirname(__file__), "" if genre is None else genre)
all_files = [os.path.join(os.path.dirname(__file__), filename) for filename in Path(root).rglob("**/*.mid*")]
print(f"Found {len(all_files)} total files")

# Filter for time signature (REMI only is 4/4 time)
filtered_files = []
failed_files = []
if os.path.exists(os.path.join(os.path.dirname(__file__), "failed_files.txt")):
    with open(os.path.join(os.path.dirname(__file__), "failed_files.txt"), 'r', encoding="utf-8") as fd:
        failed_files = set(file.strip() for file in fd.readlines())
    filtered_files = [file for file in all_files if file not in failed_files]
else:
    for filename in tqdm(all_files):
        try:
            midi_obj = miditoolkit.midi.parser.MidiFile(filename)
            if np.all(np.equal([time.denominator for time in midi_obj.time_signature_changes], 4)) and np.all(np.equal([time.numerator for time in midi_obj.time_signature_changes], 4)):
                filtered_files.append(filename)
            else:
                failed_files.append(filename)
        except:
            failed_files.append(filename)
    with open(os.path.join(os.path.dirname(__file__), "failed_files.txt"), 'w', encoding="utf-8") as fd:
        fd.write("\n".join(failed_files))
print(f"Could not parse {len(failed_files)} MIDI files...")


# Train BPE on valid MIDI files
print(f"Found {len(filtered_files)} valid MIDI files:\n{filtered_files[:5]}")
encoder = MusicEncoder(VOCAB_SIZE, load_config=True, tokenizer_config=os.path.join(os.path.dirname(__file__), "tokenizer_config.json"), use_bpe=False)
random.seed(seed)
random.shuffle(filtered_files)

# encoder.train_bpe(filtered_files)
# print(f"Trained BPE with vocab size of {len(encoder.tokenizer)}")


# Create train and validation splits
train_len = int(len(filtered_files) * (1 - val_percent))
train_files = filtered_files[:train_len]
val_files = filtered_files[train_len:]

train_dataset = encoder.tokenize_dataset(train_files)
print(f"Train has {len(train_dataset):,} tokens")
if genre == None:
    train_dataset.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
else:
    train_dataset.tofile(os.path.join(os.path.dirname(__file__), genre, "train.bin"))

val_dataset = encoder.tokenize_dataset(val_files)
print(f"Validation has {len(val_dataset):,} tokens")
if genre == None:
    val_dataset.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
else:
    val_dataset.tofile(os.path.join(os.path.dirname(__file__), genre, "val.bin"))
    