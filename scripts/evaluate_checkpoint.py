#!/usr/bin/env python3
"""Greedily decode a LibriSpeech manifest with a direct CTC checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.librispeech_csv import DatasetConstants as DS
from datasets.librispeech_csv import LibriSpeechCSVDataset
from modules.Conmamba import ConMambaCTC, ConMambaCTCConfig
from train import (
    CERScore,
    CharTokenizer,
    WERScore,
    ctc_collate,
    ctc_greedy_decode,
    ctc_valid_indices,
    get_device,
    ids_to_text,
)


def load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise SystemExit(f"checkpoint has no state_dict: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode one manifest and write paired corpus transcripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--max-utterances", type=int, default=0, help="Diagnostic cap; 0 decodes the full manifest")
    parser.add_argument("--max-duration", type=float, default=0.0, help="Optional duration cap in seconds; 0 keeps every manifest row")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    payload = load_checkpoint(checkpoint_path)
    config = payload.get("config", {})
    d_model = int(config.get("d_model", 256))
    n_blocks = int(config.get("n_blocks", 6))
    device = get_device(args.device)

    model = ConMambaCTC(
        ConMambaCTCConfig(d_model=d_model, n_blocks=n_blocks, vocab_size=29)
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()

    dataset = LibriSpeechCSVDataset(
        str(manifest_path),
        sample_rate=DS.DEFAULT_SAMPLE_RATE,
        max_duration=max(0.0, args.max_duration),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ctc_collate,
        pin_memory=False,
    )
    tokenizer = CharTokenizer()
    cer = CERScore()
    wer = WERScore()
    predictions: list[str] = []
    references: list[str] = []

    with torch.inference_mode():
        for feats, feat_lens, targets, target_lens, texts in loader:
            feats = feats.to(device)
            feat_lens = feat_lens.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)
            logits, output_lens = model(feats, feat_lens)
            valid, _, _ = ctc_valid_indices(targets, target_lens, output_lens)
            if len(valid) != len(texts):
                raise RuntimeError(
                    f"manifest contains {len(texts) - len(valid)} impossible CTC pairs"
                )
            decoded = ctc_greedy_decode(logits, output_lens)
            for token_ids, reference in zip(decoded, texts):
                if args.max_utterances and len(references) >= args.max_utterances:
                    break
                prediction = tokenizer.normalize(ids_to_text(token_ids, tokenizer))
                reference = tokenizer.normalize(reference)
                predictions.append(prediction)
                references.append(reference)
                cer.update(reference, prediction)
                wer.update(reference, prediction)
            if args.max_utterances and len(references) >= args.max_utterances:
                break

    predictions_path = Path(args.predictions)
    references_path = Path(args.references)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    references_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    references_path.write_text("\n".join(references) + "\n", encoding="utf-8")
    print(
        f"utterances={len(references)} CER={cer.cer:.6f} WER={wer.wer:.6f} "
        f"device={device}"
    )
    print(f"predictions={predictions_path}")
    print(f"references={references_path}")


if __name__ == "__main__":
    main()
