import json
from miditok import REMIPlus, TokSequence
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

class MusicEncoder():
    def __init__(self, vocab_size=50257,
                 load_config=True, 
                 tokenizer_config: Path = (os.path.join(os.path.dirname(__file__), "config.json")), 
                 use_bpe=True):
        
        self.tokenizer = REMIPlus(
            # beat_res={
            #     (0,4): 32,
            #     (4,12): 8
            # }
            # ,
            # additional_tokens={
            #     "Chord": True,
            #     "Rests": True,
            #     "Tempo": False,
            #     "TimeSignature": False,
            #     "Program": False
            # }
        )

        self.vocab_size = vocab_size
        self.tokenizer_config = tokenizer_config
        self.use_bpe = use_bpe
        if os.path.exists(tokenizer_config) and load_config:
            self.tokenizer.load_params(tokenizer_config)

        self._tokens_path = os.path.join(os.path.dirname(__file__), "tokens")
        self._bpe_tokens_path = os.path.join(os.path.dirname(__file__), "tokens_bpe")
        self._token_path_exists = os.path.exists(self._tokens_path)
        self._bpe_token_path_exists = os.path.exists(self._bpe_tokens_path)


    def _clear_dir(self, path):
        for file in [os.path.join(path, filename) for filename in os.listdir(path)]:
            os.remove(file)
        os.removedirs(path)


    def _init_dir(self, path):
        if os.path.exists(path):
            self._clear_dir(path)
        os.makedirs(path)


    def encode(self, tokens):
        seq = TokSequence(tokens=tokens)
        self.tokenizer.complete_sequence(seq)
        if self.use_bpe:
            self.tokenizer.apply_bpe(seq)
        return np.array(seq.ids, dtype=np.uint16)


    def decode(self, ids) -> TokSequence:
        seq = TokSequence(ids=ids, ids_bpe_encoded=self.use_bpe)
        if self.use_bpe:
            self.tokenizer.decode_bpe(seq)
        self.tokenizer.complete_sequence(seq)

        out_seq_tokens = []
        for i, token in enumerate(seq.tokens):
            # if token.split('_')[0] == 'Pitch' and seq.tokens[i-1].split('_')[0] != "Program": # For some reason the model won't output the program
            #     out_seq_tokens.append("Program_0")
            out_seq_tokens.append(token)

        out_seq = TokSequence(tokens=out_seq_tokens)
        self.tokenizer.complete_sequence(out_seq)

        return out_seq


    def save_to_midi(self, ids, midi_filename):
        seq = self.decode(ids)
        midi_file = self.tokenizer(np.array(seq.ids))
        midi_file.dump(midi_filename)


    def train_bpe(self, paths_to_midi=[]):
        assert self.use_bpe == True

        if not self._token_path_exists:
            self._init_dir(self._tokens_path)
            self.tokenizer.tokenize_midi_dataset(paths_to_midi, self._tokens_path)
            self._token_path_exists = True
        
        self.tokenizer.learn_bpe(vocab_size=self.vocab_size, tokens_paths=list(Path(self._tokens_path).glob("**/*.json")), start_from_empty_voc=False)
        self.tokenizer.save_params(self.tokenizer_config)
        
        self._init_dir(self._bpe_tokens_path)
        self.tokenizer.apply_bpe_to_dataset(Path(self._tokens_path), Path(self._bpe_tokens_path))
        self._bpe_token_path_exists = True


    def tokenize_dataset(self, paths_to_midi=[], use_cache=False):
        if not use_cache:
            self._init_dir(self._tokens_path)
            self.tokenizer.tokenize_midi_dataset(paths_to_midi, self._tokens_path)
            self._token_path_exists = True
            if self.use_bpe:
                self._init_dir(self._bpe_tokens_path)
                self.tokenizer.apply_bpe_to_dataset(Path(self._tokens_path), Path(self._bpe_tokens_path))
                self._bpe_token_path_exists = True

        if self.use_bpe:
            assert self._bpe_token_path_exists
            tokens_path = self._bpe_tokens_path
        else:
            assert self._token_path_exists
            tokens_path = self._tokens_path

        dataset = []
        for file in tqdm([os.path.join(tokens_path, filename) for filename in os.listdir(tokens_path) if filename[-5:] == '.json']):
            with open(file, 'r') as fd:
                dataset += [1] + json.load(fd)['ids'] + [2] # Add BOS and EOS tokens

        if not use_cache:
            self._clear_dir(self._tokens_path)
            self._token_path_exists = False
            if self.use_bpe:
                self._clear_dir(self._bpe_tokens_path)
                self._bpe_token_path_exists = False
        
        return np.array(dataset, dtype=np.uint16)
