import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import random

class LibrispeechDataset(Dataset):
    def __init__(self, root="/groups/public/LibriSpeech", manifest_path="/home/ycevan/s3prl/s3prl/data/librispeech/len_for_bucket/", 
                 sets=["train-clean-100"], bucket_size=32, max_timestep=0, debug=False):
        super(LibrispeechDataset, self).__init__()
        self.root = root
        self.manifest_path = manifest_path
        self.debug = debug
        self.bucket_size = bucket_size
        self.max_timestep = max_timestep

        # Read file
        tables = [pd.read_csv(os.path.join(manifest_path, s + ".csv")) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(
            by=["length"], ascending=False
        )
        print("[Dataset] - Training data from these sets:", str(sets))

        # Filter sequences based on max_timestep
        if max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
        elif max_timestep < 0:
            self.table = self.table[self.table.length > (-1 * max_timestep)]

        X = self.table["file_path"].tolist()
        X_lens = self.table["length"].tolist()
        self.num_samples = len(X)
        print("[Dataset] - Number of individual training instances:", self.num_samples)

        # Bucketing mechanism
        self.buckets = []
        batch_x, batch_len = [], []

        HALF_BATCHSIZE_TIME = 16000 * 16  # Define a threshold for halving batch size (adjust as needed)
        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)

            # Fill in batch_x until bucket is full
            if len(batch_x) == bucket_size:
                # Half the batch size if sequences are too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                    self.buckets.append((batch_x[:bucket_size // 2], batch_len[:bucket_size // 2]))
                    self.buckets.append((batch_x[bucket_size // 2:], batch_len[bucket_size // 2:]))
                else:
                    self.buckets.append((batch_x, batch_len))
                batch_x, batch_len = [], []

        # Gather the last batch if it has more than one sample
        if len(batch_x) > 1:
            self.buckets.append((batch_x, batch_len))

        print(f"[Dataset] - Number of buckets: {len(self.buckets)}")

    def __len__(self):
        return len(self.buckets)

    def __getitem__(self, idx):
        batch_x, batch_len = self.buckets[idx]
        waveforms = []
        lengths = []

        for file_path, length in zip(batch_x, batch_len):
            waveform, sr = torchaudio.load(os.path.join(self.root, file_path))
            if waveform.shape[0] != 1:
                waveform = waveform.mean(0, True)  # Convert to mono if needed
            waveforms.append(waveform.squeeze(0))
            lengths.append(length)

        return waveforms, lengths

def collate_fn(batch):
    # Since bucketing returns a list of (waveforms, lengths), unpack the first (and only) item
    waveforms, lengths = batch[0]
    
    # Convert lengths to tensor
    lengths = torch.tensor(lengths)
    # Pad the waveforms
    x_pad_batch = pad_sequence(waveforms, batch_first=True)
    return x_pad_batch, lengths

def get_dataloader(batch_size=32, sets=["train-clean-100"], shuffle=True, num_workers=4, debug=True, max_timestep=0):
    dataset = LibrispeechDataset(sets=sets, bucket_size=batch_size, max_timestep=max_timestep, debug=debug)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle,  # batch_size=1 since bucketing handles batching
                            num_workers=num_workers, collate_fn=collate_fn)
    if debug:
        print(f"[DEBUG] 建立 dataloader，樣本數: {len(dataset)}，bucket size: {batch_size}")
    return dataloader