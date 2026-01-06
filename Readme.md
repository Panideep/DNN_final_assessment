📘 Cross-Modal Attention Fusion for Sequential Story Generation (TensorFlow Implementation)

This repository contains an end-to-end TensorFlow implementation of a Cross-Modal Attention Fusion architecture designed to improve visual–language alignment in sequential story generation.

The model learns to encode k image–caption pairs and generate the (k+1)-th caption, using:

CNN-based image encoder

Bi-LSTM text encoder

Cross-modal attention fusion

Temporal LSTM over fused features

Caption decoder

Optional reasoning encoder

This implementation uses HuggingFace datasets, BERT tokenizer, and TensorFlow/Keras for training.

🚀 Features
✔️ Cross-Modal Attention Fusion

Fuses spatial CNN features with textual token embeddings using query–key–value attention.

✔️ Sequential Story Modeling

Temporal LSTM over fused multimodal embeddings for causal story understanding.

✔️ (Optional) Reason Encoder

Integrate explanatory or reasoning text into the decoding stage.

✔️ TensorFlow Training Loop

Fully custom training with:

GradientTape

Global norm clipping

Step-level logging via tqdm

TF Checkpoints & best-model saving

✔️ HuggingFace Dataset Support

Compatible with datasets containing sequences of images + captions (e.g. daniel3303/StoryReasoning).

📂 Project Structure
project/
│
├─ src/
│   ├─ model.py                # TensorFlow cross-modal model
│
├─ utils_tf.py                 # TF dataset builder, window generator, misc utils
├─ train.py                 # Main training script (can run from terminal)
│
├─ config.yaml                 # Dataset & model configuration
├─ requirements.txt            # Python dependencies
│
├─ README.md                   # (this file)
└─ notebook.ipynb              # Optional Jupyter notebook version

📦 Installation

Create a fresh environment (optional):

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


For Apple Silicon:

pip install tensorflow-macos tensorflow-metal

🧰 Configuration (config.yaml)

Your config.yaml controls:

Dataset (paths, sequence length, image size)

Model dimensions (embeddings, attention hidden size)

Training parameters (LR, epochs, batch size)

Dataset keys (e.g. frames, captions, reason)

Example:

dataset:
  hf_name: "daniel3303/StoryReasoning"
  seq_len: 3
  batch_size: 16
  image_size: 128
  max_caption_len: 32
  max_reason_len: 32
  frames_key: "frames"
  captions_key: "captions"
  reason_key: "reason"

model:
  image_feat_dim: 512
  image_spatial_dim: 512
  text_embed_dim: 300
  text_hidden_dim: 512
  multimodal_dim: 512
  temporal_hidden_dim: 512
  vocab_size: 30522
  pad_token_id: 0
  bos_token_id: 101
  eos_token_id: 102
  use_reason_in_fusion: false

training:
  lr: 1e-4
  epochs: 5
  device: "auto"
  log_interval: 50
  save_dir: "results_tf/checkpoints"

🧪 Training

To train the model:

python train_tf.py --config config.yaml

What training provides:

Step-level loss & perplexity logging

Epoch loss summaries

Automatic checkpoint saving

Automatic best-model weight saving

Optional validation

📈 Loss Curves

The training script collects:

Train loss per epoch

Validation loss per epoch

Train & validation perplexity

These are plotted automatically:

plt.plot(train_losses)
plt.plot(val_losses)


A loss-vs-epoch chart and optional PPL curve are displayed after training.

🧠 Model Overview
1️⃣ Image Encoder

ResNet-like CNN (or TF ConvNet) producing:

Global image feature

Spatial feature map

2️⃣ Text Encoder

Bi-LSTM over caption tokens → token-level & sentence-level embeddings.

3️⃣ Cross Modal Attention Fusion

Combines spatial visual features with linguistic features:

Q = text tokens
K, V = image patches
Attn = Softmax(QK^T / sqrt(d))


Produces fused multimodal representation per frame.

4️⃣ Temporal Encoder

LSTM over fused sequence → contextual embedding.

5️⃣ Caption Decoder

Generates the (k+1)-th caption autoregressively using LSTM.

📊 Dataset Requirements

Each HuggingFace entry must contain:

"frames" → list of PIL images

"captions" → list of caption strings

"reason" (optional) → explanatory text

Must contain at least k + 1 frames to build windows.

🧩 Customizing

You can easily modify:

Sequence length (seq_len)

Image size (128 → 224)

Add multi-head attention

Replace CNN with ViT

Replace LSTM decoder with Transformer decoder

Add contrastive loss for alignment

If you want help modifying the architecture, just ask.

🤝 Contributing

Pull requests welcome!
If you'd like additional variants (Transformer-based encoder, CLIP embeddings, ViLT-style fusion), feel free to open an issue.

📜 License

MIT License.