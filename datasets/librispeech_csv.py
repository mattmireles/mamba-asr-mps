from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
try:
    import torchaudio  # type: ignore
    HAS_TORCHAUDIO = True
except Exception:
    HAS_TORCHAUDIO = False

from utils.tokenizer import CharTokenizer


@dataclass
class LibriSpeechCSVDataset(torch.utils.data.Dataset):
    manifest: str
    sample_rate: int = 16000
    tokenizer: CharTokenizer = CharTokenizer()
    max_duration: float = 20.0

    def __post_init__(self):
        self.rows: List[Tuple[str, float, str]] = []
        p = Path(self.manifest)
        with p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                path = row[0]
                dur = float(row[1]) if row[1] else 0.0
                text = row[2]
                if self.max_duration and dur > self.max_duration:
                    continue
                self.rows.append((path, dur, text))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        wav_path, dur, text = self.rows[idx]
        if HAS_TORCHAUDIO:
            wav, sr = torchaudio.load(wav_path)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            wav = torch.mean(wav, dim=0, keepdim=False)  # mono
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=80,
                n_fft=400,
                hop_length=160,
                win_length=400,
                f_min=0,
                f_max=8000,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm=None,
                onesided=True,
            )(wav)
            mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec).transpose(0, 1)  # (T, 80)
        else:
            # Fallback: synthesize random mel frames approximating duration
            # 100 frames per second at 10ms hop
            T = max(1, int(dur * 100))
            mel_db = torch.randn(T, 80)
        tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return mel_db, torch.tensor(mel_db.shape[0]), tokens, text


def collate_fn(batch):
    feats_list, feat_lens, tokens_list, texts = zip(*batch)
    B = len(batch)
    max_T = max([f.shape[0] for f in feats_list])
    feats = torch.zeros(B, max_T, 80)
    for i, f in enumerate(feats_list):
        feats[i, : f.shape[0]] = f
    feat_lens = torch.stack(list(feat_lens))

    max_U = max([t.shape[0] for t in tokens_list])
    tokens = torch.zeros(B, max_U + 1, dtype=torch.long)  # RNNT blank prefix
    token_lens = torch.zeros(B, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        tokens[i, 1 : 1 + t.shape[0]] = t
        token_lens[i] = t.shape[0] + 1

    return feats, feat_lens, tokens, token_lens, list(texts)
