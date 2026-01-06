import os
import math
import random
import argparse
from typing import Dict, Any, List, Tuple, Generator

import numpy as np
import yaml
from tqdm import tqdm

import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import Image


from src.model import CrossModalStoryModelTF  # <-- TF version of the model


# ---------------------------------------------------
#  Utils
# ---------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(device_str: str) -> str:
    """
    TensorFlow automatically uses available GPU.
    This function is just for logging / compatibility.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if device_str == "auto":
        if len(gpus) > 0:
            return "gpu"
        return "cpu"
    return device_str


# ---------------------------------------------------
#  Dataset: Build k → k+1 story windows (indices)
# ---------------------------------------------------
def build_window_indices(hf_dataset, seq_len: int, frames_key: str) -> List[Tuple[int, int]]:
    """
    Precompute list of (story_index, start_t) windows such that:

      - frames[start_t : start_t + seq_len] are inputs
      - frame at index (start_t + seq_len) provides target caption
    """
    indices: List[Tuple[int, int]] = []
    k = seq_len
    for story_idx in range(len(hf_dataset)):
        item = hf_dataset[story_idx]
        frames = item[frames_key]
        num_frames = len(frames)
        if num_frames < k + 1:
            continue
        for start_t in range(0, num_frames - (k + 1) + 1):
            indices.append((story_idx, start_t))

    print(f"[Dataset] Built {len(indices)} windows from {len(hf_dataset)} stories.")
    return indices


# ---------------------------------------------------
#  Generator: yield one window at a time (for tf.data)
# ---------------------------------------------------
def window_generator(
    hf_dataset,
    indices: List[Tuple[int, int]],
    cfg: Dict[str, Any],
    tokenizer: AutoTokenizer,
    use_reason: bool = False,
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Python generator that yields examples:

      images:          (T, H, W, 3) float32 in [0,1]
      caption_ids:     (T, L_in)    int32
      caption_mask:    (T, L_in)    int32
      tgt_caption_ids: (L_out,)     int32
      [reason_ids]:    (Lr,)        int32
      [reason_mask]:   (Lr,)        int32
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
        frames = item[frames_key]        # list of PIL images / arrays
        captions = item[captions_key]    # list of strings

        # ---- k input frames & captions ----
        imgs = []
        caption_ids_list = []
        caption_mask_list = []

        for t in range(start_t, start_t + seq_len):
            img = frames[t]
            if not isinstance(img, Image.Image):
                # HF image feature can be dict or array; convert if needed
                img = Image.fromarray(np.array(img))

            img = img.resize((image_size, image_size))
            img_np = np.array(img).astype("float32")
            if img_np.ndim == 2:  # grayscale -> RGB
                img_np = np.stack([img_np] * 3, axis=-1)
            if img_np.shape[-1] == 4:  # RGBA -> RGB
                img_np = img_np[..., :3]
            img_np /= 255.0
            imgs.append(img_np)  # (H, W, 3)

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

        images_arr = np.stack(imgs, axis=0)                   # (T, H, W, 3)
        caption_ids_arr = np.stack(caption_ids_list, axis=0)  # (T, L)
        caption_mask_arr = np.stack(caption_mask_list, axis=0)  # (T, L)

        # ---- target caption = frame at start_t + k ----
        target_caption_text = captions[start_t + seq_len]
        tgt_enc = tokenizer(
            target_caption_text,
            padding="max_length",
            truncation=True,
            max_length=max_caption_len,
            return_tensors="np",
        )
        tgt_ids = tgt_enc["input_ids"].squeeze(0).astype("int32")  # (L,)

        example = {
            "images": images_arr,
            "caption_ids": caption_ids_arr,
            "caption_mask": caption_mask_arr,
            "tgt_caption_ids": tgt_ids,
        }

        # ---- optional reason / explanation ----
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
#  Build tf.data.Dataset from generator
# ---------------------------------------------------
def build_tf_dataset(
    hf_dataset,
    cfg: Dict[str, Any],
    tokenizer: AutoTokenizer,
    use_reason: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    dataset_cfg = cfg["dataset"]
    seq_len = dataset_cfg["seq_len"]
    max_caption_len = dataset_cfg["max_caption_len"]
    max_reason_len = dataset_cfg["max_reason_len"]
    image_size = dataset_cfg["image_size"]

    frames_key = dataset_cfg.get("frames_key", "frames")

    indices = build_window_indices(hf_dataset, seq_len, frames_key)

    # base output signature
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
        ds = ds.shuffle(buffer_size=min(4096, len(indices)))

    return ds, len(indices)


# ---------------------------------------------------
#  Training / Evaluation
# ---------------------------------------------------
def train_one_epoch(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    steps: int,
    optimizer: tf.keras.optimizers.Optimizer,
    pad_token_id: int,
    log_interval: int,
    epoch: int,
):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    running_loss = 0.0
    total_tokens = 0

    # progress bar
    pbar = tqdm(dataset, total=steps, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        images = batch["images"]              # (B, T, H, W, 3)
        caption_ids = batch["caption_ids"]    # (B, T, L)
        caption_mask = batch["caption_mask"]  # (B, T, L)
        tgt_caption_ids = batch["tgt_caption_ids"]  # (B, L_out)

        reason_ids = batch.get("reason_ids", None)
        reason_mask = batch.get("reason_mask", None)

        with tf.GradientTape() as tape:
            logits, _ = model(
                images=images,
                caption_ids=caption_ids,
                tgt_caption_ids=tgt_caption_ids,
                caption_mask=caption_mask,
                reason_ids=reason_ids,
                reason_mask=reason_mask,
                training=True,
            )  # logits: (B, L_out, V)

            # loss per position
            loss_per_pos = loss_fn(
                tgt_caption_ids, logits
            )  # (B, L_out)

            mask = tf.cast(tf.not_equal(tgt_caption_ids, pad_token_id), tf.float32)
            loss_masked = loss_per_pos * mask  # (B, L_out)

            batch_loss_sum = tf.reduce_sum(loss_masked)  # scalar
            batch_tokens = tf.reduce_sum(mask)           # scalar

            loss = batch_loss_sum / tf.maximum(batch_tokens, 1.0)

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        running_loss += batch_loss_sum.numpy()
        total_tokens += batch_tokens.numpy()

        if (step + 1) % log_interval == 0 or (step + 1) == steps:
            avg_loss = running_loss / max(total_tokens, 1)
            ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                ppl=f"{ppl:.2f}",
                step=f"{step+1}/{steps}",
            )

        if step + 1 >= steps:
            break

    epoch_loss = running_loss / max(total_tokens, 1)
    epoch_ppl = math.exp(epoch_loss) if epoch_loss < 20 else float("inf")
    print(f"[Train] Epoch {epoch+1} Completed  Loss: {epoch_loss:.4f}  PPL: {epoch_ppl:.2f}")
    return epoch_loss, epoch_ppl


def evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    steps: int,
    pad_token_id: int,
    split_name: str = "Val",
):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    running_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataset, total=steps, desc=f"Evaluating ({split_name})")
    for step, batch in enumerate(pbar):
        images = batch["images"]
        caption_ids = batch["caption_ids"]
        caption_mask = batch["caption_mask"]
        tgt_caption_ids = batch["tgt_caption_ids"]

        reason_ids = batch.get("reason_ids", None)
        reason_mask = batch.get("reason_mask", None)

        logits, _ = model(
            images=images,
            caption_ids=caption_ids,
            tgt_caption_ids=tgt_caption_ids,
            caption_mask=caption_mask,
            reason_ids=reason_ids,
            reason_mask=reason_mask,
            training=False,
        )

        loss_per_pos = loss_fn(tgt_caption_ids, logits)  # (B, L_out)
        mask = tf.cast(tf.not_equal(tgt_caption_ids, pad_token_id), tf.float32)
        loss_masked = loss_per_pos * mask

        batch_loss_sum = tf.reduce_sum(loss_masked).numpy()
        batch_tokens = tf.reduce_sum(mask).numpy()

        running_loss += batch_loss_sum
        total_tokens += batch_tokens

        if step + 1 >= steps:
            break

    avg_loss = running_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    print(f"[{split_name}] Loss: {avg_loss:.4f}  PPL: {ppl:.2f}")
    return avg_loss, ppl


# ---------------------------------------------------
#  Main
# ---------------------------------------------------
def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    set_seed(42)

    train_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    device_str = get_device(train_cfg.get("device", "auto"))
    print(f"Using device: {device_str}")
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))

    # HF tokenizer (BERT-base-uncased)
    tokenizer_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load HF dataset
    hf_name = dataset_cfg["hf_name"]
    print(f"Loading dataset: {hf_name}")
    ds_dict = load_dataset(hf_name)

    train_split_name = dataset_cfg.get("train_split", "train")
    val_split_name = dataset_cfg.get("val_split", "validation")

    hf_train = ds_dict[train_split_name]
    hf_val = ds_dict[val_split_name] if val_split_name in ds_dict else None

    use_reason = model_cfg.get("use_reason_in_fusion", False)

    # Build tf.data.Dataset
    train_ds, train_windows = build_tf_dataset(
        hf_dataset=hf_train,
        cfg=cfg,
        tokenizer=tokenizer,
        use_reason=use_reason,
        shuffle=True,
    )

    if hf_val is not None:
        val_ds, val_windows = build_tf_dataset(
            hf_dataset=hf_val,
            cfg=cfg,
            tokenizer=tokenizer,
            use_reason=use_reason,
            shuffle=False,
        )
    else:
        val_ds, val_windows = None, 0

    batch_size = dataset_cfg["batch_size"]

    train_steps = 50
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if val_ds is not None:
        val_steps = 50
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_steps = 0

    # Build model
    model = CrossModalStoryModelTF(cfg)

    # Optimizer
    lr = train_cfg.get("lr", 1e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    epochs = train_cfg.get("epochs", 5)
    log_interval = train_cfg.get("log_interval", 50)
    save_dir = train_cfg.get("save_dir", "results/checkpoints_tf")
    os.makedirs(save_dir, exist_ok=True)

    pad_token_id = model_cfg["pad_token_id"]

    # Checkpointing
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=save_dir, max_to_keep=3
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss, train_ppl = train_one_epoch(
            model=model,
            dataset=train_ds,
            steps=train_steps,
            optimizer=optimizer,
            pad_token_id=pad_token_id,
            log_interval=log_interval,
            epoch=epoch,
        )

        if val_ds is not None and val_steps > 0:
            val_loss, val_ppl = evaluate(
                model=model,
                dataset=val_ds,
                steps=val_steps,
                pad_token_id=pad_token_id,
                split_name="Val",
            )
        else:
            val_loss, val_ppl = train_loss, train_ppl

        # Save checkpoint every epoch
        ckpt_path = ckpt_manager.save()
        print(f"Saved checkpoint to {ckpt_path}")

        # Track best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, "best_model_best")
            model.save_weights(best_path)
            print(f"New best model weights saved to {best_path} (Val loss: {val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    main(args.config)
