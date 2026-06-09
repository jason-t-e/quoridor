"""
Model Registry for Final SINN.

Manages all checkpoint I/O, metadata tracking, and champion promotion.
Ensures training never starts from scratch unless explicitly requested,
and gameplay always uses the strongest stable model.
"""

import os
import json
import shutil
import torch
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger("QuoridorAI")

CHECKPOINT_DIR = "data/checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
METADATA_PATH = os.path.join(CHECKPOINT_DIR, "metadata.json")
CURRENT_MODEL_PREFIX = "current_v"


def _ensure_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------

def load_metadata() -> Dict[str, Any]:
    """Load the global metadata.json, or return a fresh default."""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "champion_path": None,
        "champion_version": None,
        "champion_elo": 1200.0,
        "champion_games_played": 0,
        "training_history": [],
        "total_training_steps": 0,
        "total_games_played": 0,
    }


def save_metadata(meta: Dict[str, Any]):
    _ensure_dirs()
    tmp = METADATA_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    # Atomic replace (Windows: os.replace is atomic on NTFS)
    os.replace(tmp, METADATA_PATH)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    training_step: int = 0,
    games_played: int = 0,
    elo: float = 1200.0,
    version: str = "v0",
    config: Optional[dict] = None,
    path: Optional[str] = None,
) -> str:
    """
    Save a full checkpoint with rich metadata.
    Returns the path it was saved to.
    """
    _ensure_dirs()
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, f"{CURRENT_MODEL_PREFIX}{version}.pt")

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_step": training_step,
        "games_played": games_played,
        "timestamp": _now_iso(),
        "elo": elo,
        "version": version,
        "config": config or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    logger.info(f"Checkpoint saved: {path} (v{version}, step={training_step}, games={games_played}, elo={elo:.1f})")
    return path


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint dict.  Handles both old-format and new-format files.
    Returns the raw dict (caller decides what to do with it).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Normalise old-format checkpoints that lack metadata fields
    if "training_step" not in ckpt:
        ckpt.setdefault("training_step", 0)
    if "games_played" not in ckpt:
        ckpt.setdefault("games_played", 0)
    if "timestamp" not in ckpt:
        ckpt.setdefault("timestamp", _now_iso())
    if "elo" not in ckpt:
        ckpt.setdefault("elo", 1200.0)
    if "version" not in ckpt:
        ckpt.setdefault("version", "v0")
    if "config" not in ckpt:
        ckpt.setdefault("config", {})
    # Old files may store weights directly instead of under model_state_dict
    if "model_state_dict" not in ckpt:
        # Assume the entire dict is a state_dict
        ckpt = {
            "model_state_dict": ckpt,
            "optimizer_state_dict": None,
            "training_step": 0,
            "games_played": 0,
            "timestamp": _now_iso(),
            "elo": 1200.0,
            "version": "v0",
            "config": {},
        }
    return ckpt


# ---------------------------------------------------------------------------
# Champion (best_model) management
# ---------------------------------------------------------------------------

def load_best_model(device: Optional[torch.device] = None):
    """
    Load the champion model.
    Returns (model, metadata_dict) or None if no champion exists.
    """
    if not os.path.exists(BEST_MODEL_PATH):
        return None

    from models.quoridor_net import QuoridorNet

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(BEST_MODEL_PATH, device=device)
    model = QuoridorNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict" and k != "optimizer_state_dict"}
    logger.info(
        f"Champion loaded: v{ckpt['version']}, elo={ckpt['elo']:.1f}, "
        f"games={ckpt['games_played']}, step={ckpt['training_step']}"
    )
    return model, meta


def promote_to_champion(source_path: str):
    """
    Atomically promote a checkpoint to champion (best_model.pt).
    Updates metadata.json accordingly.
    """
    _ensure_dirs()
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Cannot promote: {source_path} does not exist")

    # Atomic copy
    tmp = BEST_MODEL_PATH + ".tmp"
    shutil.copy2(source_path, tmp)
    os.replace(tmp, BEST_MODEL_PATH)

    # Update metadata
    ckpt = load_checkpoint(source_path)
    meta = load_metadata()
    meta["champion_path"] = BEST_MODEL_PATH
    meta["champion_version"] = ckpt["version"]
    meta["champion_elo"] = ckpt["elo"]
    meta["champion_games_played"] = ckpt["games_played"]
    meta["training_history"].append({
        "version": ckpt["version"],
        "elo": ckpt["elo"],
        "games": ckpt["games_played"],
        "training_step": ckpt["training_step"],
        "timestamp": ckpt["timestamp"],
        "promoted_at": _now_iso(),
    })
    save_metadata(meta)

    logger.info(f"Champion promoted: v{ckpt['version']} (elo={ckpt['elo']:.1f})")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def load_or_init_for_training(
    device: Optional[torch.device] = None,
    lr: float = 0.001,
    from_scratch: bool = False,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
    """
    Load the best checkpoint for continued training, or initialise fresh.
    Returns (model, optimizer, metadata_dict).
    """
    from models.quoridor_net import QuoridorNet
    import torch.optim as optim

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not from_scratch and os.path.exists(BEST_MODEL_PATH):
        ckpt = load_checkpoint(BEST_MODEL_PATH, device=device)
        model = QuoridorNet().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if ckpt.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")
        meta = {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "optimizer_state_dict")}
        logger.info(f"Resumed training from champion v{meta['version']} (step={meta['training_step']}, games={meta['games_played']})")
        return model, optimizer, meta

    # Fresh init
    logger.info("Initialising new model from scratch")
    model = QuoridorNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    meta = {
        "training_step": 0,
        "games_played": 0,
        "elo": 1200.0,
        "version": "v1",
        "timestamp": _now_iso(),
        "config": {},
    }
    return model, optimizer, meta


def next_version(current_version: str) -> str:
    """Increment version string: 'v3' -> 'v4'."""
    try:
        num = int(current_version.lstrip("v"))
        return f"v{num + 1}"
    except ValueError:
        return "v1"


def list_checkpoints() -> List[Dict[str, Any]]:
    """List all checkpoints with their metadata."""
    _ensure_dirs()
    results = []
    for f in sorted(os.listdir(CHECKPOINT_DIR)):
        if not (f.endswith(".pt") or f.endswith(".pth")):
            continue
        path = os.path.join(CHECKPOINT_DIR, f)
        try:
            ckpt = load_checkpoint(path)
            results.append({
                "path": path,
                "filename": f,
                "version": ckpt.get("version", "?"),
                "elo": ckpt.get("elo", 0),
                "games_played": ckpt.get("games_played", 0),
                "training_step": ckpt.get("training_step", 0),
                "timestamp": ckpt.get("timestamp", ""),
                "is_champion": (f == "best_model.pt"),
            })
        except Exception:
            results.append({"path": path, "filename": f, "error": "unreadable"})
    return results


# ---------------------------------------------------------------------------
# Migration helper — upgrade old-format checkpoints
# ---------------------------------------------------------------------------

def migrate_legacy_checkpoints():
    """
    If best_model.pt does not exist but old-format .pt files do,
    pick the most recent one and promote it as champion v1.
    """
    if os.path.exists(BEST_MODEL_PATH):
        return  # Already have a champion

    _ensure_dirs()
    candidates = []
    for f in os.listdir(CHECKPOINT_DIR):
        if (f.endswith(".pt") or f.endswith(".pth")) and f != "best_model.pt":
            full = os.path.join(CHECKPOINT_DIR, f)
            candidates.append((os.path.getmtime(full), full))

    if not candidates:
        return  # Nothing to migrate

    candidates.sort(reverse=True)  # Most recent first
    best_legacy = candidates[0][1]

    logger.info(f"Migrating legacy checkpoint {best_legacy} -> champion v1")

    # Load, re-save with proper metadata, then promote
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(best_legacy, device=device)

    from models.quoridor_net import QuoridorNet
    import torch.optim as optim

    model = QuoridorNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if ckpt.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass

    path = save_checkpoint(
        model, optimizer,
        training_step=ckpt.get("training_step", 0),
        games_played=ckpt.get("games_played", 0),
        elo=1200.0,
        version="v1",
    )
    promote_to_champion(path)
