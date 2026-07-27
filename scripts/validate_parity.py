#!/usr/bin/env python3
"""Validate direct CTC parity across PyTorch full/chunked and Core ML chunked."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import coremltools as ct  # type: ignore
import numpy as np
import torch
import torchaudio  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.librispeech_csv import waveform_to_log_mel
from modules.Conmamba import ConMambaCTC, ConMambaCTCConfig


def resolve_contract(mlpackage: Path, explicit: str) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            mlpackage.parent / "contract.json",
            mlpackage.with_suffix(".contract.json"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise SystemExit(f"contract not found; checked: {rendered}")


def package_sha256(package_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in package_path.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(package_path)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_contract(path: Path, mlpackage: Path) -> Dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schemaVersion") != 1:
        raise SystemExit(
            f"unsupported contract schemaVersion={contract.get('schemaVersion')!r}"
        )
    if contract.get("modelType") != "ctc29":
        raise SystemExit(f"expected modelType=ctc29, got {contract.get('modelType')!r}")
    if contract["model"]["vocabSize"] != 29 or len(contract["vocab"]) != 29:
        raise SystemExit("contract must contain exactly 29 CTC symbols")
    recorded_hash = contract.get("packageSHA256")
    if recorded_hash:
        actual_hash = package_sha256(mlpackage)
        if recorded_hash != actual_hash:
            raise SystemExit(
                "Core ML package hash does not match contract: "
                f"expected {recorded_hash}, got {actual_hash}"
            )
    return contract


def resolve_checkpoint(contract: Dict, contract_path: Path, explicit: str) -> Path:
    candidate = (
        Path(explicit)
        if explicit
        else contract_path.parent / contract["referenceCheckpoint"]
    )
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise SystemExit(f"reference checkpoint not found: {candidate}")
    return candidate


def load_checkpoint(path: Path) -> Dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"checkpoint must contain a dictionary: {path}")
    return payload


def load_model(checkpoint: Path, contract: Dict) -> ConMambaCTC:
    payload = load_checkpoint(checkpoint)
    config = contract["model"]
    model = ConMambaCTC(
        ConMambaCTCConfig(
            d_model=int(config["dModel"]),
            n_blocks=int(config["nBlocks"]),
            vocab_size=int(config["vocabSize"]),
        )
    ).eval().float().cpu()
    state_dict = payload.get("state_dict", payload.get("model_state"))
    if not isinstance(state_dict, dict):
        raise SystemExit(f"checkpoint has no state_dict/model_state: {checkpoint}")
    model.load_state_dict(state_dict, strict=True)
    return model


def prepare_features(wav_path: Path, chunk_frames: int, chunks: int) -> Tuple[torch.Tensor, bool]:
    waveform, sample_rate = torchaudio.load(str(wav_path))
    features = waveform_to_log_mel(waveform, sample_rate).float().cpu()
    required_frames = chunk_frames * chunks
    repeated = features.shape[0] < required_frames
    if repeated:
        repeats = math.ceil(required_frames / features.shape[0])
        features = features.repeat((repeats, 1))
    return features[:required_frames].unsqueeze(0).contiguous(), repeated


def greedy_tokens(logits: np.ndarray) -> List[int]:
    ids = logits.argmax(axis=-1).reshape(-1).tolist()
    collapsed: List[int] = []
    previous = None
    for token in ids:
        if token != previous and token != 0:
            collapsed.append(int(token))
        previous = token
    return collapsed


def render_tokens(tokens: Sequence[int], vocab: Sequence[str]) -> str:
    return "".join(vocab[token] for token in tokens)


def correlation(expected: np.ndarray, actual: np.ndarray) -> float:
    left = expected.astype(np.float64, copy=False).reshape(-1)
    right = actual.astype(np.float64, copy=False).reshape(-1)
    if left.size != right.size:
        raise ValueError(f"shape size mismatch: {left.size} != {right.size}")
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def metrics(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    return correlation(expected, actual), float(
        np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)))
    )


def chunks(tensor: torch.Tensor, chunk_frames: int) -> Iterable[torch.Tensor]:
    for start in range(0, tensor.shape[1], chunk_frames):
        yield tensor[:, start : start + chunk_frames]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate PyTorch/Core ML parity for the CTC streaming contract"
    )
    parser.add_argument("--mlpackage", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--contract", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--corr-min", type=float, default=0.999)
    parser.add_argument("--max-error", type=float, default=1e-3)
    args = parser.parse_args()

    mlpackage = Path(args.mlpackage).resolve()
    wav_path = Path(args.wav).resolve()
    if not mlpackage.is_dir():
        raise SystemExit(f"Core ML package not found: {mlpackage}")
    if not wav_path.is_file():
        raise SystemExit(f"WAV not found: {wav_path}")
    if args.chunks < 3:
        raise SystemExit("--chunks must be at least 3")

    contract_path = resolve_contract(mlpackage, args.contract)
    contract = load_contract(contract_path, mlpackage)
    checkpoint = resolve_checkpoint(contract, contract_path, args.checkpoint)
    model = load_model(checkpoint, contract)

    chunk_frames = int(contract["streaming"]["chunkFrames"])
    features, repeated = prepare_features(wav_path, chunk_frames, args.chunks)
    context_frames = int(contract["streaming"]["convContextFrames"])
    state_shape = tuple(int(value) for value in contract["io"]["stateShape"])
    expected_audio_shape = tuple(int(value) for value in contract["io"]["audioShape"])
    probe_audio = torch.cat(
        [
            torch.zeros(1, context_frames, features.shape[2]),
            features[:, :chunk_frames],
        ],
        dim=1,
    )
    if tuple(probe_audio.shape) != expected_audio_shape:
        raise SystemExit(
            "feature shape does not match contract: "
            f"{tuple(probe_audio.shape)} != {expected_audio_shape}"
        )

    with torch.no_grad():
        full_logits, _ = model(
            features,
            torch.tensor([features.shape[1]], dtype=torch.long),
        )
        torch_state = torch.zeros(state_shape, dtype=torch.float32)
        torch_context = torch.zeros(
            1,
            context_frames,
            features.shape[2],
            dtype=torch.float32,
        )
        torch_logits_parts: List[torch.Tensor] = []
        torch_states: List[np.ndarray] = []
        for feature_chunk in chunks(features, chunk_frames):
            model_input = torch.cat([torch_context, feature_chunk], dim=1)
            logits, torch_state = model.streaming_forward(model_input, torch_state)
            torch_logits_parts.append(logits)
            torch_states.append(torch_state.numpy().copy())
            torch_context = model_input[:, -context_frames:, :]
        torch_chunked_logits = torch.cat(torch_logits_parts, dim=1).numpy()

    coreml_model = ct.models.MLModel(
        str(mlpackage),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    coreml_state = np.zeros(state_shape, dtype=np.float32)
    coreml_context = np.zeros(
        (1, context_frames, features.shape[2]),
        dtype=np.float32,
    )
    coreml_logits_parts: List[np.ndarray] = []
    failures: List[str] = []
    io = contract["io"]

    print(f"fixture={wav_path}")
    print(
        f"frames={features.shape[1]} chunks={args.chunks} "
        f"chunk_frames={chunk_frames} fixture_repeated={str(repeated).lower()}"
    )
    for index, feature_chunk in enumerate(chunks(features, chunk_frames), start=1):
        coreml_audio = np.concatenate(
            [coreml_context, feature_chunk.numpy()],
            axis=1,
        ).astype(np.float32)
        prediction = coreml_model.predict(
            {
                io["audioInput"]: coreml_audio,
                io["stateInput"]: coreml_state,
            }
        )
        coreml_logits = np.asarray(prediction[io["logitsOutput"]], dtype=np.float32)
        coreml_state = np.asarray(prediction[io["stateOutput"]], dtype=np.float32)
        coreml_logits_parts.append(coreml_logits)
        coreml_context = coreml_audio[:, -context_frames:, :]

        logits_corr, logits_error = metrics(
            torch_logits_parts[index - 1].numpy(),
            coreml_logits,
        )
        state_corr, state_error = metrics(torch_states[index - 1], coreml_state)
        print(
            f"chunk={index} logits_corr={logits_corr:.9f} "
            f"logits_max_error={logits_error:.9g} "
            f"state_corr={state_corr:.9f} state_max_error={state_error:.9g}"
        )
        if logits_corr < args.corr_min or logits_error > args.max_error:
            failures.append(f"chunk {index} logits outside tolerance")
        if state_corr < args.corr_min or state_error > args.max_error:
            failures.append(f"chunk {index} state outside tolerance")

    coreml_chunked_logits = np.concatenate(coreml_logits_parts, axis=1)
    vocab = contract["vocab"]
    full_text = render_tokens(greedy_tokens(full_logits.numpy()), vocab)
    torch_chunked_text = render_tokens(greedy_tokens(torch_chunked_logits), vocab)
    coreml_text = render_tokens(greedy_tokens(coreml_chunked_logits), vocab)
    boundary_equal = full_text == torch_chunked_text
    parity_equal = torch_chunked_text == coreml_text
    total_corr, total_error = metrics(torch_chunked_logits, coreml_chunked_logits)

    print(
        f"aggregate_logits_corr={total_corr:.9f} "
        f"aggregate_logits_max_error={total_error:.9g}"
    )
    print(f"pytorch_full_transcript={full_text!r}")
    print(f"pytorch_chunked_transcript={torch_chunked_text!r}")
    print(f"coreml_chunked_transcript={coreml_text!r}")
    print(f"full_vs_chunked_transcript_equal={str(boundary_equal).lower()}")
    print(f"pytorch_vs_coreml_transcript_equal={str(parity_equal).lower()}")

    if total_corr < args.corr_min or total_error > args.max_error:
        failures.append("aggregate logits outside tolerance")
    if not boundary_equal:
        failures.append(
            "full-vs-chunked transcript differs; implement convolution context"
        )
    if not parity_equal:
        failures.append("PyTorch-vs-Core ML greedy transcript differs")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("PASS: numerical parity and greedy transcript gates satisfied")


if __name__ == "__main__":
    main()
