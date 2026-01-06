import math
from typing import Optional, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers, Model


# -----------------------------
#  Image Encoder (TF / Keras)
# -----------------------------
class ImageEncoderTF(layers.Layer):
    """
    CNN-based image encoder.
    Uses ResNet50 backbone and returns:
      - global_feats: (B, T, image_feat_dim)
      - spatial_feats: (B, T, num_patches, image_spatial_dim)

    Expected input:
      images: (B, T, H, W, 3)  (channels_last)
    """

    def __init__(self, image_feat_dim: int = 512, image_spatial_dim: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.image_feat_dim = image_feat_dim
        self.image_spatial_dim = image_spatial_dim

        # ResNet50 backbone without top classifier, no pooling
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            pooling=None,
        )
        self.backbone = base_model

        # 1x1 conv (implemented as Dense over channels) to project to desired spatial dim
        self.spatial_proj = layers.Conv2D(
            filters=self.image_spatial_dim,
            kernel_size=1,
            padding="same",
            activation=None,
        )

        # Global pooling + projection to image_feat_dim
        self.global_pool = layers.GlobalAveragePooling2D()
        self.global_proj = layers.Dense(self.image_feat_dim)

    def call(self, images, training=False):
        """
        images: (B, T, H, W, 3)

        Returns:
          global_feats:  (B, T, D)
          spatial_feats: (B, T, N, D_spatial)  where N = H'*W'
        """
        # Merge batch and time
        B = tf.shape(images)[0]
        T = tf.shape(images)[1]
        H = tf.shape(images)[2]
        W = tf.shape(images)[3]

        x = tf.reshape(images, (B * T, H, W, 3))  # (B*T, H, W, 3)

        # ResNet features: (B*T, H', W', C_resnet)
        feat_map = self.backbone(x, training=training)

        # Project spatial channels
        feat_map = self.spatial_proj(feat_map)  # (B*T, H', W', image_spatial_dim)
        B_T = tf.shape(feat_map)[0]
        Hp = tf.shape(feat_map)[1]
        Wp = tf.shape(feat_map)[2]
        D_spatial = tf.shape(feat_map)[3]

        # Global pooled feature
        global_feats = self.global_pool(feat_map)              # (B*T, D_spatial)
        global_feats = self.global_proj(global_feats)          # (B*T, image_feat_dim)
        global_feats = tf.reshape(global_feats, (B, T, self.image_feat_dim))

        # Spatial features: flatten H',W' -> patches
        spatial_feats = tf.reshape(feat_map, (B_T, Hp * Wp, D_spatial))  # (B*T, N, D_spatial)
        spatial_feats = tf.reshape(spatial_feats, (B, T, Hp * Wp, D_spatial))  # (B, T, N, D_spatial)

        return global_feats, spatial_feats


# -----------------------------
#  Text Encoder (for k captions)
# -----------------------------
class TextEncoderTF(layers.Layer):
    """
    Bi-LSTM text encoder over tokens for each caption.

    input_ids:    (B, T, L)  or (B, L)
    attention_mask: (B, T, L) or (B, L), 1 = keep, 0 = pad

    Returns:
      token_feats: (B, T, L, hidden_dim) or (B, L, hidden_dim)
      sent_feats:  (B, T, hidden_dim)    or (B, hidden_dim)
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=False)
        self.hidden_dim = hidden_dim
        # BiLSTM with hidden_dim total (hidden_dim/2 each direction)
        self.bilstm = layers.Bidirectional(
            layers.LSTM(
                hidden_dim // 2,
                return_sequences=True,
                return_state=False,
            ),
            merge_mode="concat",
        )

    def call(self, input_ids, attention_mask=None, training=False):
        x = input_ids
        if len(x.shape) == 3:
            # (B, T, L) -> (B*T, L)
            B = tf.shape(x)[0]
            T = tf.shape(x)[1]
            L = tf.shape(x)[2]
            x_flat = tf.reshape(x, (B * T, L))
            if attention_mask is not None:
                mask_flat = tf.reshape(attention_mask, (B * T, L))
            else:
                mask_flat = None
        else:
            B = tf.shape(x)[0]
            L = tf.shape(x)[1]
            T = None
            x_flat = x
            mask_flat = attention_mask

        emb = self.embedding(x_flat)  # (B*T, L, E) or (B, L, E)

        outputs = self.bilstm(emb, training=training)  # (B*T, L, hidden_dim)

        if mask_flat is not None:
            mask = tf.cast(mask_flat, tf.float32)[:, :, tf.newaxis]  # (B*T, L, 1)
            summed = tf.reduce_sum(outputs * mask, axis=1)           # (B*T, hidden_dim)
            lengths = tf.reduce_sum(mask, axis=1)                    # (B*T, 1)
            lengths = tf.clip_by_value(lengths, 1e-6, tf.float32.max)
            sent_feats = summed / lengths
        else:
            sent_feats = tf.reduce_mean(outputs, axis=1)             # (B*T, hidden_dim)

        if T is not None:
            token_feats = tf.reshape(outputs, (B, T, L, self.hidden_dim))
            sent_feats = tf.reshape(sent_feats, (B, T, self.hidden_dim))
        else:
            token_feats = outputs  # (B, L, hidden_dim)

        return token_feats, sent_feats


# -----------------------------
#  (Optional) Reason Encoder
# -----------------------------
class ReasonEncoderTF(layers.Layer):
    """
    Separate encoder for reasoning / explanation text.

    reason_ids:   (B, Lr)
    reason_mask:  (B, Lr)
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=False)
        self.hidden_dim = hidden_dim
        self.bilstm = layers.Bidirectional(
            layers.LSTM(
                hidden_dim // 2,
                return_sequences=True,
                return_state=False,
            ),
            merge_mode="concat",
        )

    def call(self, reason_ids, reason_mask=None, training=False):
        emb = self.embedding(reason_ids)                         # (B, Lr, E)
        outputs = self.bilstm(emb, training=training)           # (B, Lr, hidden_dim)

        if reason_mask is not None:
            mask = tf.cast(reason_mask, tf.float32)[:, :, tf.newaxis]  # (B, Lr, 1)
            summed = tf.reduce_sum(outputs * mask, axis=1)
            lengths = tf.reduce_sum(mask, axis=1)
            lengths = tf.clip_by_value(lengths, 1e-6, tf.float32.max)
            reason_feat = summed / lengths
        else:
            reason_feat = tf.reduce_mean(outputs, axis=1)        # (B, hidden_dim)

        return reason_feat


# -----------------------------
#  Cross-Modal Attention Fusion
# -----------------------------
class CrossModalAttentionFusionTF(layers.Layer):
    """
    Cross-modal attention between image spatial features and text token features
    for each time step.

    img_spatial_feats: (B, T, N, Di)
    txt_token_feats:   (B, T, L, Dt)
    txt_sent_feats:    (B, T, Dt)
    txt_mask:          (B, T, L)  (1=keep, 0=pad)

    Returns:
      fused: (B, T, multimodal_dim)
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        multimodal_dim: int,
        attn_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.img_key = layers.Dense(attn_dim)
        self.img_value = layers.Dense(attn_dim)
        self.txt_query = layers.Dense(attn_dim)

        self.txt_proj = layers.Dense(attn_dim)

        self.attn_dropout = layers.Dropout(dropout)
        self.fuse_proj = layers.Dense(multimodal_dim, activation="tanh")
        self.fuse_ln = layers.LayerNormalization()

        self.attn_dim = attn_dim
        self.multimodal_dim = multimodal_dim

    def call(self, img_spatial_feats, txt_token_feats, txt_sent_feats, txt_mask=None, training=False):
        B = tf.shape(img_spatial_feats)[0]
        T = tf.shape(img_spatial_feats)[1]
        N = tf.shape(img_spatial_feats)[2]
        Di = tf.shape(img_spatial_feats)[3]

        L = tf.shape(txt_token_feats)[2]
        Dt = tf.shape(txt_token_feats)[3]

        img = tf.reshape(img_spatial_feats, (B * T, N, Di))   # (B*T, N, Di)
        txt = tf.reshape(txt_token_feats, (B * T, L, Dt))     # (B*T, L, Dt)
        sent = tf.reshape(txt_sent_feats, (B * T, Dt))        # (B*T, Dt)

        if txt_mask is not None:
            mask = tf.reshape(txt_mask, (B * T, L))           # (B*T, L)
        else:
            mask = None

        # Projections
        K = self.img_key(img)                                 # (B*T, N, A)
        V = self.img_value(img)                               # (B*T, N, A)
        Q = self.txt_query(txt)                               # (B*T, L, A)
        txt_global = self.txt_proj(sent)                      # (B*T, A)

        # Attention: Q * K^T
        attn_scores = tf.matmul(Q, K, transpose_b=True)       # (B*T, L, N)
        attn_scores = attn_scores / tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))

        attn_weights = tf.nn.softmax(attn_scores, axis=-1)    # (B*T, L, N)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attended_img = tf.matmul(attn_weights, V)             # (B*T, L, A)

        # Mean pool over tokens with optional mask
        if mask is not None:
            m = tf.cast(mask, tf.float32)[:, :, tf.newaxis]   # (B*T, L, 1)
            summed = tf.reduce_sum(attended_img * m, axis=1)  # (B*T, A)
            lengths = tf.reduce_sum(m, axis=1)                # (B*T, 1)
            lengths = tf.clip_by_value(lengths, 1e-6, tf.float32.max)
            attended_img_global = summed / lengths
        else:
            attended_img_global = tf.reduce_mean(attended_img, axis=1)  # (B*T, A)

        # Combine with global text
        fused = tf.concat([attended_img_global, txt_global], axis=-1)  # (B*T, 2A)
        fused = self.fuse_proj(fused)                                  # (B*T, M)
        fused = self.fuse_ln(fused)

        fused = tf.reshape(fused, (B, T, self.multimodal_dim))         # (B, T, M)
        return fused


# -----------------------------
#  Temporal Encoder (over time)
# -----------------------------
class TemporalEncoderTF(layers.Layer):
    """
    LSTM over time dimension for fused multimodal features.

    x: (B, T, M)
    Returns:
      outputs: (B, T, H)
      context: (B, H) (last hidden)
    """

    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.lstm = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,
        )
        self.hidden_dim = hidden_dim

    def call(self, x, training=False):
        outputs, h_n, c_n = self.lstm(x, training=training)   # outputs: (B, T, H)
        context = h_n                                         # (B, H)
        return outputs, context


# -----------------------------
#  Caption Decoder (k+1)
# -----------------------------
class CaptionDecoderTF(layers.Layer):
    """
    LSTM decoder for generating the (k+1)-th caption.

    Forward (teacher forcing):
      tgt_ids:   (B, L_out)
      context_vec: (B, context_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pad_idx: int,
        bos_idx: int,
        eos_idx: int,
        context_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=False)
        self.lstm = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,
        )
        self.init_h = layers.Dense(hidden_dim, activation="tanh")
        self.init_c = layers.Dense(hidden_dim, activation="tanh")
        self.out_proj = layers.Dense(vocab_size)

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def call(self, tgt_ids, context_vec, training=False):
        # Teacher forcing
        emb = self.embedding(tgt_ids)                         # (B, L_out, E)

        h0 = self.init_h(context_vec)                         # (B, H)
        c0 = self.init_c(context_vec)                         # (B, H)

        outputs, _, _ = self.lstm(emb, initial_state=[h0, c0], training=training)
        logits = self.out_proj(outputs)                       # (B, L_out, V)

        return logits

    def generate(self, context_vec, max_len: int = 32):
        """
        Greedy generation.
        context_vec: (B, context_dim)
        Returns:
          generated_ids: (B, max_len)
        """
        B = tf.shape(context_vec)[0]

        h = self.init_h(context_vec)                          # (B, H)
        c = self.init_c(context_vec)                          # (B, H)

        # Start with BOS
        inputs = tf.fill((B, 1), tf.cast(self.bos_idx, tf.int32))  # (B, 1)
        generated = []

        for _ in range(max_len):
            emb = self.embedding(inputs)                      # (B, 1, E)
            outputs, h, c = self.lstm(emb, initial_state=[h, c])
            logits = self.out_proj(outputs[:, -1, :])         # (B, V)
            next_tokens = tf.argmax(logits, axis=-1, output_type=tf.int32)  # (B,)
            generated.append(next_tokens[:, tf.newaxis])
            inputs = next_tokens[:, tf.newaxis]

        generated_ids = tf.concat(generated, axis=1)          # (B, max_len)
        return generated_ids


# -----------------------------
#  Full Cross-Modal Story Model (TF)
# -----------------------------
class CrossModalStoryModelTF(Model):
    """
    End-to-end TF model for:
      - encoding k image–caption pairs with cross-modal attention fusion
      - temporal encoding over sequence
      - decoding the (k+1)-th caption

    images:         (B, T, H, W, 3)
    caption_ids:    (B, T, L_in)
    tgt_caption_ids:(B, L_out)
    caption_mask:   (B, T, L_in)
    reason_ids:     (B, Lr) (optional)
    reason_mask:    (B, Lr) (optional)
    """

    def __init__(self, cfg: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)

        model_cfg = cfg["model"]
        vocab_size = model_cfg["vocab_size"]
        pad_idx = model_cfg["pad_token_id"]
        bos_idx = model_cfg["bos_token_id"]
        eos_idx = model_cfg["eos_token_id"]

        self.image_encoder = ImageEncoderTF(
            image_feat_dim=model_cfg["image_feat_dim"],
            image_spatial_dim=model_cfg["image_spatial_dim"],
        )

        self.text_encoder = TextEncoderTF(
            vocab_size=vocab_size,
            embed_dim=model_cfg["text_embed_dim"],
            hidden_dim=model_cfg["text_hidden_dim"],
            pad_idx=pad_idx,
        )

        self.use_reason = model_cfg.get("use_reason_in_fusion", False)
        if self.use_reason:
            self.reason_encoder = ReasonEncoderTF(
                vocab_size=vocab_size,
                embed_dim=model_cfg["reason_embed_dim"],
                hidden_dim=model_cfg["reason_hidden_dim"],
                pad_idx=pad_idx,
            )
            reason_out_dim = model_cfg["reason_hidden_dim"]
        else:
            self.reason_encoder = None
            reason_out_dim = 0

        self.cross_modal_fusion = CrossModalAttentionFusionTF(
            image_dim=model_cfg["image_spatial_dim"],
            text_dim=model_cfg["text_hidden_dim"],
            multimodal_dim=model_cfg["multimodal_dim"],
            attn_dim=model_cfg["cross_modal_attn_dim"],
            dropout=model_cfg["cross_modal_dropout"],
        )

        self.temporal_encoder = TemporalEncoderTF(
            input_dim=model_cfg["multimodal_dim"],
            hidden_dim=model_cfg["temporal_hidden_dim"],
        )

        context_dim = model_cfg["temporal_hidden_dim"] + reason_out_dim

        self.decoder = CaptionDecoderTF(
            vocab_size=vocab_size,
            embed_dim=model_cfg["text_decoder_embed_dim"],
            hidden_dim=model_cfg["text_decoder_hidden"],
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            context_dim=context_dim,
        )

    def call(
        self,
        images,
        caption_ids,
        tgt_caption_ids,
        caption_mask=None,
        reason_ids=None,
        reason_mask=None,
        training=False,
    ):
        """
        Forward pass with teacher forcing.
        """
        # 1) Image encoding
        img_global_feats, img_spatial_feats = self.image_encoder(images, training=training)

        # 2) Text encoding
        txt_token_feats, txt_sent_feats = self.text_encoder(
            caption_ids,
            attention_mask=caption_mask,
            training=training,
        )

        # 3) Cross-modal fusion over time
        fused_feats = self.cross_modal_fusion(
            img_spatial_feats=img_spatial_feats,
            txt_token_feats=txt_token_feats,
            txt_sent_feats=txt_sent_feats,
            txt_mask=caption_mask,
            training=training,
        )

        # 4) Temporal encoder
        temporal_outputs, temporal_context = self.temporal_encoder(fused_feats, training=training)

        # 5) Optional reason encoder
        if self.use_reason and (reason_ids is not None):
            reason_feat = self.reason_encoder(reason_ids, reason_mask, training=training)
            context_vec = tf.concat([temporal_context, reason_feat], axis=-1)
        else:
            context_vec = temporal_context

        # 6) Decode (k+1) caption
        logits = self.decoder(tgt_caption_ids, context_vec, training=training)

        extra = {
            "img_global_feats": img_global_feats,
            "img_spatial_feats": img_spatial_feats,
            "txt_sent_feats": txt_sent_feats,
            "fused_feats": fused_feats,
            "temporal_outputs": temporal_outputs,
            "temporal_context": temporal_context,
            "context_vec": context_vec,
        }

        return logits, extra

    def generate_next_caption(
        self,
        images,
        caption_ids,
        caption_mask=None,
        reason_ids=None,
        reason_mask=None,
        max_len: int = 32,
    ):
        """
        Encode k image-caption pairs and greedily generate the (k+1)-th caption.

        Returns:
          generated_ids: (B, max_len)
        """
        img_global_feats, img_spatial_feats = self.image_encoder(images, training=False)

        txt_token_feats, txt_sent_feats = self.text_encoder(
            caption_ids,
            attention_mask=caption_mask,
            training=False,
        )

        fused_feats = self.cross_modal_fusion(
            img_spatial_feats=img_spatial_feats,
            txt_token_feats=txt_token_feats,
            txt_sent_feats=txt_sent_feats,
            txt_mask=caption_mask,
            training=False,
        )

        _, temporal_context = self.temporal_encoder(fused_feats, training=False)

        if self.use_reason and (reason_ids is not None):
            reason_feat = self.reason_encoder(reason_ids, reason_mask, training=False)
            context_vec = tf.concat([temporal_context, reason_feat], axis=-1)
        else:
            context_vec = temporal_context

        generated_ids = self.decoder.generate(context_vec, max_len=max_len)
        return generated_ids
