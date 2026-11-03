"""Resolve pretrained model weights from the local ``models/`` directory.

All HuggingFace-format weights live in ``<repo>/models/<model-name>`` (e.g.
``models/Molmo-7B-D-0924``). Set ``T_FUNS3D_MODELS_ROOT`` to use another root.
Models are never downloaded: a missing directory raises an error.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = Path(os.environ.get("T_FUNS3D_MODELS_ROOT", REPO_ROOT / "models"))

MOLMO_MODEL = "Molmo-7B-D-0924"
SAM_MODEL = "sam-vit-huge"
FGCLIP_MODEL = "fg-clip-large"
LLM_MODEL = "Qwen3-14B"


def resolve_model_path(name_or_path: str) -> str:
    """Map a model name, HF repo id (``org/name``) or path to a local directory."""
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return str(path.resolve())

    local = MODELS_ROOT / Path(name_or_path).name
    if local.is_dir():
        return str(local)

    raise FileNotFoundError(
        f"Model '{name_or_path}' not found locally (looked for {local}). "
        f"Put the weights under {MODELS_ROOT} or set T_FUNS3D_MODELS_ROOT."
    )
