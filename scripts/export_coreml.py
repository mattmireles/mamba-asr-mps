#!/usr/bin/env python3
"""Export the direct 29-logit ConMamba CTC streaming contract to Core ML."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("MAMBA_DISABLE_RECORD_FUNCTION", "1")

import numpy as np
import torch
import torch.nn as nn

try:
    import coremltools as ct  # type: ignore
except Exception as exc:  # pragma: no cover - exercised by full-stack installs
    raise SystemExit(f"coremltools is required for export: {exc}")

from datasets.librispeech_csv import DatasetConstants
from modules.Conmamba import AudioConstants, ConMambaCTC, ConMambaCTCConfig
from utils.tokenizer import CharTokenizer


DEFAULT_CHUNK_FRAMES = int(os.environ.get("MAMBA_CHUNK_DEFAULT", "256"))
DEFAULT_D_MODEL = 256
DEFAULT_N_BLOCKS = 6
SCHEMA_VERSION = 1


class StreamingCTCWrapper(nn.Module):
    """Flat tensor-I/O wrapper consumed by Core ML and the Swift runner."""

    def __init__(self, model: ConMambaCTC):
        super().__init__()
        self.model = model

    def forward(
        self,
        audio_chunk: torch.Tensor,
        mamba_states_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.streaming_forward(audio_chunk, mamba_states_in)


def _checkpoint_payload(path: Path) -> Dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {path}")
    return payload


def load_ctc_model(
    checkpoint: Optional[Path],
    d_model: int,
    n_blocks: int,
    seed: int,
) -> Tuple[ConMambaCTC, Dict, Optional[Path]]:
    """Load trained weights or create a deterministic random reference model."""
    checkpoint_config: Dict = {}
    if checkpoint is not None:
        payload = _checkpoint_payload(checkpoint)
        raw_config = payload.get("config", {})
        checkpoint_config = raw_config if isinstance(raw_config, dict) else {}
        d_model = int(checkpoint_config.get("d_model", d_model))
        n_blocks = int(checkpoint_config.get("n_blocks", n_blocks))
    else:
        torch.manual_seed(seed)

    config = ConMambaCTCConfig(
        d_model=d_model,
        n_blocks=n_blocks,
        vocab_size=29,
    )
    model = ConMambaCTC(config).eval().float().cpu()

    if checkpoint is not None:
        payload = _checkpoint_payload(checkpoint)
        state_dict = payload.get("state_dict", payload.get("model_state"))
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"checkpoint has no direct ConMamba state_dict/model_state: {checkpoint}"
            )
        model.load_state_dict(state_dict, strict=True)

    return model, checkpoint_config, checkpoint


def _relative_reference(reference: Path, contract_path: Path) -> str:
    return os.path.relpath(reference.resolve(), contract_path.parent.resolve())


def write_random_reference(
    model: ConMambaCTC,
    output_path: Path,
    seed: int,
) -> Path:
    """Persist the exact random weights used by the parity gate."""
    reference_path = output_path.with_suffix(".reference.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "d_model": model.cfg.d_model,
                "n_blocks": model.cfg.n_blocks,
                "vocab_size": model.cfg.vocab_size,
            },
            "seed": seed,
        },
        reference_path,
    )
    return reference_path


def build_contract(
    model: ConMambaCTC,
    chunk_frames: int,
    reference_path: Path,
    contract_path: Path,
    precision: str,
) -> Dict:
    """Build the single source of truth read by Python and Swift."""
    tokenizer = CharTokenizer()
    vocab = [""] + [tokenizer.id_to_char[index] for index in range(1, 29)]
    output_frames = chunk_frames
    for _ in range(2):
        output_frames = (output_frames + 1) // 2
    state_shape = [
        model.cfg.n_blocks,
        1,
        model.cfg.d_model,
        AudioConstants.MAMBA_STATE_DIM,
    ]
    mel = DatasetConstants.get_mel_config()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "modelType": "ctc29",
        "precision": precision,
        "referenceCheckpoint": _relative_reference(reference_path, contract_path),
        "model": {
            "dModel": model.cfg.d_model,
            "nBlocks": model.cfg.n_blocks,
            "stateDim": AudioConstants.MAMBA_STATE_DIM,
            "vocabSize": model.cfg.vocab_size,
            "timeReduction": AudioConstants.TOTAL_SUBSAMPLING_FACTOR,
        },
        "streaming": {
            "chunkFrames": chunk_frames,
            "outputFrames": output_frames,
            "boundaryPolicy": "causal-conv-left-context-carry-mamba",
            "convContextFrames": AudioConstants.CONV_CONTEXT_FRAMES,
        },
        "mel": {
            "sampleRate": mel["sample_rate"],
            "nFFT": mel["n_fft"],
            "winLength": mel["win_length"],
            "hopLength": mel["hop_length"],
            "nMels": mel["n_mels"],
            "fMin": mel["f_min"],
            "fMax": mel["f_max"],
            "center": mel["center"],
            "power": mel["power"],
            "melScale": mel["mel_scale"],
            "norm": mel["norm"],
            "logScale": "natural",
            "logFloor": DatasetConstants.LOG_FLOOR,
            "window": "hann_periodic",
        },
        "vocab": vocab,
        "io": {
            "audioInput": "audio_chunk",
            "stateInput": "mamba_states_in",
            "logitsOutput": "logits",
            "stateOutput": "mamba_states_out",
            "audioShape": [
                1,
                chunk_frames + AudioConstants.CONV_CONTEXT_FRAMES,
                DatasetConstants.N_MELS,
            ],
            "stateShape": state_shape,
            "logitsShape": [1, output_frames, model.cfg.vocab_size],
        },
    }


def package_sha256(package_path: Path) -> str:
    """Hash package files deterministically for traceable validation output."""
    digest = hashlib.sha256()
    for path in sorted(p for p in package_path.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(package_path)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def export_ctc_model(
    model: ConMambaCTC,
    output_path: Path,
    contract_path: Path,
    reference_path: Path,
    chunk_frames: int,
    use_fp16: bool,
    compute_units: str,
) -> None:
    """Trace, convert, save, and describe one fixed-shape CTC package."""
    state_shape = (
        model.cfg.n_blocks,
        1,
        model.cfg.d_model,
        AudioConstants.MAMBA_STATE_DIM,
    )
    example_audio = torch.zeros(
        1,
        chunk_frames + AudioConstants.CONV_CONTEXT_FRAMES,
        DatasetConstants.N_MELS,
    )
    example_states = torch.zeros(state_shape)
    traced = torch.jit.trace(
        StreamingCTCWrapper(model).eval(),
        (example_audio, example_states),
        strict=False,
    )

    units = {
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu-ne": getattr(ct.ComputeUnit, "CPU_AND_NE", ct.ComputeUnit.ALL),
    }[compute_units]
    precision = ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32
    array_dtype = np.float16 if use_fp16 else np.float32

    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="audio_chunk",
                shape=example_audio.shape,
                dtype=array_dtype,
            ),
            ct.TensorType(
                name="mamba_states_in",
                shape=example_states.shape,
                dtype=array_dtype,
            ),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=array_dtype),
            ct.TensorType(name="mamba_states_out", dtype=array_dtype),
        ],
        convert_to="mlprogram",
        compute_precision=precision,
        compute_units=units,
        minimum_deployment_target=ct.target.iOS16,
    )

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output_path))

    contract = build_contract(
        model,
        chunk_frames,
        reference_path,
        contract_path,
        "fp16" if use_fp16 else "fp32",
    )
    contract["packageSHA256"] = package_sha256(output_path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    print(f"Core ML model saved: {output_path}")
    print(f"Contract saved: {contract_path}")
    print(f"Reference checkpoint: {reference_path}")
    print(f"Package SHA-256: {contract['packageSHA256']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a direct 29-logit streaming CTC Core ML package"
    )
    parser.add_argument("--checkpoint", "--model", dest="checkpoint", default="")
    parser.add_argument(
        "--output",
        default="exports/MambaASR_ctc29.mlpackage",
    )
    parser.add_argument("--contract", default="")
    parser.add_argument("--chunk", "--chunk_length", dest="chunk", type=int, default=DEFAULT_CHUNK_FRAMES)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n-blocks", type=int, default=DEFAULT_N_BLOCKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpu", "cpu-gpu", "cpu-ne"],
        default="cpu",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    contract_path = (
        Path(args.contract).resolve()
        if args.contract
        else output_path.parent / "contract.json"
    )
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if args.chunk <= 0 or args.chunk % AudioConstants.TOTAL_SUBSAMPLING_FACTOR:
        raise SystemExit("--chunk must be positive and divisible by 4")

    model, _, source_checkpoint = load_ctc_model(
        checkpoint,
        args.d_model,
        args.n_blocks,
        args.seed,
    )
    reference_path = source_checkpoint or write_random_reference(
        model,
        output_path,
        args.seed,
    )
    export_ctc_model(
        model,
        output_path,
        contract_path,
        reference_path,
        args.chunk,
        args.fp16,
        args.compute_units,
    )


if __name__ == "__main__":
    main()
