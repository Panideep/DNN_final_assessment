import os
import random
from typing import Dict, Any, List, Tuple, Generator

import numpy as np
import yaml

import tensorflow as tf
from datasets import Dataset as HFDataset  # only for type hints (optional)
from PIL import Image


# ---------------------------------------------------
#  General helpers
# ---------------------------------------------------
def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML config file.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(device_str: str) -> str:
    """
    Return a simple device descriptor ("cpu" / "gpu") according to config.
    TensorFlow will automatically place ops on GPU if available.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if device_str == "auto":
        return "gpu" if len(gpus) > 0 else "cpu"
    return device_str


# ---------------------------------------------------
#  Dataset: build k → k+1 window indices
# ---------------------------------------------------
def build_window_indices(hf_dataset, seq_len: int, frames_key: str) -> List[Tuple[int, int]]:
    """
    Build sliding windows of length (seq_len + 1) per story:

      [t, t+1, ..., t+k-1]  -> inputs
      [t+k]                 -> target caption

    Returns a list of (story_index, start_t).
    """
    indices: List[Tuple[int, int]] = []
    k = seq_len

    for story_idx in range(len(hf_dataset)):
        item = hf_dataset[story_idx]
        frames = item[frames_key]

        # We assume frames is a list; adapt here if different
        num_frames = len(frames)

        if num_frames < k + 1:
            continue

        # Sliding windows
        for start_t in range(0, num_frames - (k + 1) + 1):
            indices.append((story_idx, start_t))

    print(f"[Dataset] Built {len(indices)} windows from {len(hf_dataset)} stories.")
    return indices


# ---------------------------------------------------
#  Per-window generator (for tf.data)
# ---------------------------------------------------
def window_generator(
    hf_dataset,
    indices: List[Tuple[int, int]],
    cfg: Dict[str, Any],
    tokenizer,
    use_reason: bool = False,
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Python generator that yields windows of shape:

      images:          (T, H, W, 3)  float32 in [0,1]
      caption_ids:     (T, L_in)     int32
      caption_mask:    (T, L_in)     int32
      tgt_caption_ids: (L_out,)      int32
      [reason_ids]:    (Lr,)         int32
      [reason_mask]:   (Lr,)         int32

    This mirrors the behaviour of the PyTorch StoryReasoningWindowDataset.
    """
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    seq_len = dataset_cfg["seq_len"]
    max_caption_len = dataset_cfg["max_caption_len"]
    max_reason_len = dataset_cfg["max_reason_len"]
    image_size = dataset_cfg["image_size"]

    frames_key = dataset_cfg.get("frames_key", "frames")
    captions_key = dataset_cfg.get("captions_key", "captions")
    reason_key = dataset_cfg.get("reason_key", "reason")

    pad_token_id = model_cfg["pad_token_id"]

    for (story_idx, start_t) in indices:
        item = hf_dataset[story_idx]

        frames = item[frames_key]        # list of images
        captions = item[captions_key]    # list of caption strings

        # ------------- k input frames & captions -------------

        imgs = []
        caption_ids_list = []
        caption_mask_list = []

        for t in range(start_t, start_t + seq_len):
            img = frames[t]

            # HF image is often PIL.Image; otherwise convert via np.array -> PIL
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))

            # Resize and convert to float32 [0,1], channels_last
            img = img.resize((image_size, image_size))
            img_np = np.array(img).astype("float32")

            # Ensure RGB
            if img_np.ndim == 2:  # grayscale -> RGB
                img_np = np.stack([img_np] * 3, axis=-1)
            if img_np.shape[-1] == 4:  # RGBA -> RGB
                img_np = img_np[..., :3]

            img_np /= 255.0
            imgs.append(img_np)  # (H, W, 3)

            # Tokenize caption
            cap_text = captions[t]
            enc = tokenizer(
                cap_text,
                padding="max_length",
                truncation=True,
                max_length=max_caption_len,
                return_tensors="np",
            )
            cap_ids = enc["input_ids"].squeeze(0).astype("int32")        # (L,)
            cap_mask = enc["attention_mask"].squeeze(0).astype("int32")  # (L,)
            caption_ids_list.append(cap_ids)
            caption_mask_list.append(cap_mask)

        images_arr = np.stack(imgs, axis=0)                    # (T, H, W, 3)
        caption_ids_arr = np.stack(caption_ids_list, axis=0)   # (T, L)
        caption_mask_arr = np.stack(caption_mask_list, axis=0) # (T, L)

        # ------------- target caption = frame at index start_t + k -------------
        target_caption_text = captions[start_t + seq_len]
        tgt_enc = tokenizer(
            target_caption_text,
            padding="max_length",
            truncation=True,
            max_length=max_caption_len,
            return_tensors="np",
        )
        tgt_ids = tgt_enc["input_ids"].squeeze(0).astype("int32")  # (L,)

        example: Dict[str, np.ndarray] = {
            "images": images_arr,
            "caption_ids": caption_ids_arr,
            "caption_mask": caption_mask_arr,
            "tgt_caption_ids": tgt_ids,
        }

        # ------------- optional reasoning text -------------
        if use_reason and reason_key in item and item[reason_key] is not None:
            reason_text = item[reason_key]
            r_enc = tokenizer(
                reason_text,
                padding="max_length",
                truncation=True,
                max_length=max_reason_len,
                return_tensors="np",
            )
            reason_ids = r_enc["input_ids"].squeeze(0).astype("int32")        # (Lr,)
            reason_mask = r_enc["attention_mask"].squeeze(0).astype("int32")  # (Lr,)
            example["reason_ids"] = reason_ids
            example["reason_mask"] = reason_mask

        yield example


# ---------------------------------------------------
#  Build tf.data.Dataset (TF equivalent of Dataset+collate_fn)
# ---------------------------------------------------
def build_tf_dataset(
    hf_dataset,
    cfg: Dict[str, Any],
    tokenizer,
    use_reason: bool = False,
    shuffle: bool = False,
) -> Tuple[tf.data.Dataset, int]:
    """
    Create a tf.data.Dataset that yields batches compatible with CrossModalStoryModelTF.

    Returned dataset (after .batch) will provide:

      images:          (B, T, H, W, 3)
      caption_ids:     (B, T, L)
      caption_mask:    (B, T, L)
      tgt_caption_ids: (B, L_out)
      [reason_ids]:    (B, Lr)
      [reason_mask]:   (B, Lr)
    """
    dataset_cfg = cfg["dataset"]

    seq_len = dataset_cfg["seq_len"]
    max_caption_len = dataset_cfg["max_caption_len"]
    max_reason_len = dataset_cfg["max_reason_len"]
    image_size = dataset_cfg["image_size"]

    frames_key = dataset_cfg.get("frames_key", "frames")

    # Build the list of window indices
    indices = build_window_indices(hf_dataset, seq_len, frames_key)

    # Base output signature for a single element (no batch dimension yet)
    out_sig = {
        "images": tf.TensorSpec(
            shape=(seq_len, image_size, image_size, 3), dtype=tf.float32
        ),
        "caption_ids": tf.TensorSpec(
            shape=(seq_len, max_caption_len), dtype=tf.int32
        ),
        "caption_mask": tf.TensorSpec(
            shape=(seq_len, max_caption_len), dtype=tf.int32
        ),
        "tgt_caption_ids": tf.TensorSpec(
            shape=(max_caption_len,), dtype=tf.int32
        ),
    }

    if use_reason:
        out_sig["reason_ids"] = tf.TensorSpec(
            shape=(max_reason_len,), dtype=tf.int32
        )
        out_sig["reason_mask"] = tf.TensorSpec(
            shape=(max_reason_len,), dtype=tf.int32
        )

    # Generator wrapper
    gen = lambda: window_generator(
        hf_dataset=hf_dataset,
        indices=indices,
        cfg=cfg,
        tokenizer=tokenizer,
        use_reason=use_reason,
    )

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=out_sig,
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(4096, len(indices)),
            reshuffle_each_iteration=True,
        )

    return ds, len(indices)
