import os
import random
import yaml
from typing import List, Tuple

import numpy as np
from PIL import Image

import tensorflow as tf

# Try to reuse implementations from src.train where applicable
try:
    from src.train import build_tf_dataset as _build_tf_dataset_from_src
    from src.train import set_seed as _set_seed_from_src
    from src.train import load_config as _load_config_from_src
    from src.train import get_device as _get_device_from_src
except Exception:
    _build_tf_dataset_from_src = None
    _set_seed_from_src = None
    _load_config_from_src = None
    _get_device_from_src = None


# ------------------
# Lightweight helpers
# ------------------

def set_seed(seed: int = 42):
    """Set Python / NumPy / TF seeds for reproducibility."""
    if _set_seed_from_src is not None:
        try:
            return _set_seed_from_src(seed)
        except Exception:
            pass
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_config(path: str = "config.yaml"):
    """Load a YAML config file. Returns a dict."""
    if _load_config_from_src is not None:
        try:
            return _load_config_from_src(path)
        except Exception:
            pass
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_str: str = "auto") -> str:
    """Return a simple device string: 'gpu' or 'cpu' for logging/compatibility."""
    if _get_device_from_src is not None:
        try:
            return _get_device_from_src(device_str)
        except Exception:
            pass
    gpus = tf.config.list_physical_devices("GPU")
    if device_str == "auto":
        return "gpu" if len(gpus) > 0 else "cpu"
    return device_str


# ------------------
# Notebook-oriented builder
# ------------------

def _resize_image_pil(img, image_size: int):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.array(img).astype("uint8")
    return arr


def build_tf_datasets_from_examples(
    examples: List[dict],
    config: dict,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Build simple `tf.data.Dataset` objects from an `examples` list created in the notebook.

    Each `example` is expected to be: {'images': [PIL.Image,...] (len=seq_len), 'caption_tokens': [int,...]}

    Returns: (train_ds, val_ds, (n_train, n_val)) where each dataset yields `(inputs, targets)`
    and `inputs` is a dict with keys `images_seq` and `caption_tokens`.
    """
    set_seed(seed)

    dataset_cfg = config.get("dataset", {})
    seq_len = int(dataset_cfg.get("seq_len", 3))
    batch_size = int(dataset_cfg.get("batch_size", 8))
    image_size = int(dataset_cfg.get("image_size", 128))
    max_caption_len = int(dataset_cfg.get("max_caption_len", 32))

    # Shuffle and split indices
    n = len(examples)
    indices = list(range(n))
    random.shuffle(indices)
    n_val = int(n * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    def gen_from_indices(idxs):
        for i in idxs:
            ex = examples[i]
            imgs = ex.get("images")
            if imgs is None:
                continue
            # Ensure seq_len frames
            if len(imgs) < seq_len:
                imgs = imgs + [imgs[-1]] * (seq_len - len(imgs))
            else:
                imgs = imgs[:seq_len]

            imgs_arr = np.stack([_resize_image_pil(im, image_size) for im in imgs], axis=0).astype("uint8")

            # caption tokens: may already include BOS/EOS; pad/truncate to max_caption_len
            cap = ex.get("caption_tokens", [])
            cap = list(map(int, cap))
            if len(cap) < max_caption_len:
                cap = cap + [config.get("model", {}).get("pad_token_id", 0)] * (max_caption_len - len(cap))
            else:
                cap = cap[:max_caption_len]

            cap_arr = np.array(cap).astype("int32")

            inputs = {
                "images_seq": imgs_arr,  # (T, H, W, C) uint8
                "caption_tokens": cap_arr,  # (L,) int32
            }
            targets = cap_arr.copy()  # simple target mirror
            yield inputs, targets

    # Create tf.data datasets
    output_signature = (
        {
            "images_seq": tf.TensorSpec(shape=(seq_len, image_size, image_size, 3), dtype=tf.uint8),
            "caption_tokens": tf.TensorSpec(shape=(max_caption_len,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(max_caption_len,), dtype=tf.int32),
    )

    train_ds = tf.data.Dataset.from_generator(lambda: gen_from_indices(train_idx), output_signature=output_signature)
    val_ds = tf.data.Dataset.from_generator(lambda: gen_from_indices(val_idx), output_signature=output_signature)

    # Batch and prefetch
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, (len(train_idx), len(val_idx))


# Provide simple alias for notebooks that expect `build_tf_dataset`
def build_tf_dataset(*args, **kwargs):
    """Compatibility wrapper: prefer using `build_tf_dataset` from `src.train` for HF datasets.
    For the notebook-level quick examples you can call `build_tf_datasets_from_examples`.
    """
    # Try dynamic import at call time to avoid import-time failures (e.g., package path issues)
    try:
        from src.train import build_tf_dataset as _dyn_build
        return _dyn_build(*args, **kwargs)
    except Exception:
        # If caller passed an `examples` list (not an HF dataset object), offer the alternative
        if len(args) > 0 and isinstance(args[0], list):
            # assume signature: (examples, config, ...)
            try:
                return build_tf_datasets_from_examples(*args, **kwargs)
            except Exception as e:
                raise RuntimeError("Failed to build datasets from examples list: " + str(e)) from e

        raise NotImplementedError(
            "build_tf_dataset from `src.train` could not be imported. "
            "Ensure `src` is on PYTHONPATH or run the notebook from the project root. "
            "If you have an `examples` list (created in the notebook), call `build_tf_datasets_from_examples(examples, config, ...)` instead."
        )
