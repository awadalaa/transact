import tensorflow as tf
import tensorflow_models as tfm
from transact.transact_config import TransActConfig


class TensorflowTransAct(tf.keras.Model):
    def __init__(self, transact_config: TransActConfig, training=False):
        super().__init__()
        self.transact_config = transact_config
        self.training = training
        self.action_vocab: list = self.transact_config.action_vocab

        if self.transact_config.concat_candidate_emb:
            transformer_in_dim = (
                self.transact_config.action_emb_dim
                + self.transact_config.item_emb_dim * 2
            )
        else:
            transformer_in_dim = (
                self.transact_config.action_emb_dim + self.transact_config.item_emb_dim
            )

        self.action_type_lookup = tf.keras.layers.IntegerLookup(
            num_oov_indices=1,
            mask_token=-1,
            vocabulary=list(self.transact_config.action_vocab),
            output_mode="int",
        )
        self.action_emb_module = tf.keras.layers.Embedding(
            input_dim=len(self.action_vocab) + 1 + 1,  # oov
            output_dim=self.transact_config.action_emb_dim,
            input_length=self.transact_config.seq_len,
        )

        self.transformer_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=self.transact_config.num_layer,
            num_attention_heads=self.transact_config.nhead,
            intermediate_size=self.transact_config.dim_feedforward,
            activation="relu",
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            use_bias=False,
            norm_first=True,
            norm_epsilon=1e-06,
            intermediate_dropout=0.0,
        )

        if self.transact_config.concat_max_pool:
            self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
            self.out_linear = tf.keras.layers.Dense(transformer_in_dim)

    def call(
        self,
        action_type_seq: tf.Tensor,
        item_embedding_seq: tf.Tensor,
        action_time_seq: tf.Tensor,
        request_time: tf.Tensor,
        item_embedding: tf.Tensor,
    ):
        # step 1: get the latest N actions from sequence features
        action_type_seq = action_type_seq[:, : self.transact_config.seq_len]
        item_embedding_seq = item_embedding_seq[:, : self.transact_config.seq_len, :]
        action_time_seq = action_time_seq[:, : self.transact_config.seq_len]

        # step 2: get action embedding
        action_type_seq = self.action_type_lookup(action_type_seq)
        action_emb_tensor = self.action_emb_module(action_type_seq)

        # step 3: create mask that tells transformer which position to be ignored by the attention.
        key_padding_mask = tf.math.less_equal(action_type_seq, 0)

        # mask actions that happened in a time window before the request time
        request_time = tf.expand_dims(request_time, axis=-1)
        request_time = tf.broadcast_to(request_time, tf.shape(action_time_seq))

        # randomly sample a time window to introduce randomness
        rand_time_window_ms = tf.random.uniform(
            shape=(),
            minval=0,
            maxval=self.transact_config.time_window_ms,
            dtype=tf.int32,
        )
        short_time_window_idx_trn = tf.math.less(
            request_time - action_time_seq, rand_time_window_ms
        )
        short_time_window_idx_eval = tf.math.less(request_time - action_time_seq, 0)

        # adjust the mask accordingly
        if self.training:
            key_padding_mask = self._adjust_mask(
                key_padding_mask, short_time_window_idx_trn
            )
        else:
            key_padding_mask = self._adjust_mask(
                key_padding_mask, short_time_window_idx_eval
            )

        # step 4: concat seq embedding with action embedding and candidate embedding
        action_pin_emb = tf.concat((action_emb_tensor, item_embedding_seq), axis=2)

        if self.transact_config.concat_candidate_emb:
            item_embedding_expanded = tf.expand_dims(item_embedding, axis=1)
            item_embedding_expanded = tf.tile(
                item_embedding_expanded, multiples=[1, self.seq_len, 1]
            )
            action_pin_emb = tf.concat(
                (action_pin_emb, item_embedding_expanded), axis=-1
            )

        key_padding_mask = tf.expand_dims(key_padding_mask, axis=-1)
        tfmr_out = self.transformer_encoder(
            encoder_inputs=action_pin_emb, attention_mask=key_padding_mask
        )

        # output_concat = []
        # if self.transact_config.concat_max_pool:
        # Apply max pooling to the transformer output
        pooled_max = self.max_pool(tfmr_out)
        pooled_out = self.out_linear(pooled_max)
        # output_concat.append(pooled_out)

        # if self.transact_config.latest_n_emb > 0:
        #     tfmr_out = tfmr_out[:, :self.transact_config.latest_n_emb]
        # output_concat.append(tf.reshape(tfmr_out, [tf.shape(tfmr_out)[0], -1]))
        # output = tf.keras.layers.Concatenate(axis=1)(output_concat)
        # return output
        return pooled_out

    def _adjust_mask(self, mask, short_time_window_idx):
        # Make sure not all actions in the sequence are masked
        mask = tf.cast(mask, tf.int64)
        short_time_window_idx = tf.cast(short_time_window_idx, tf.int64)

        mask = tf.bitwise.bitwise_or(mask, short_time_window_idx)

        # Set the first column of the mask to zeros
        mask = tf.concat([tf.zeros_like(mask[:, :1]), mask[:, 1:]], axis=1)

        # Create a new attention mask with -inf for masked positions
        new_attn_mask = tf.where(tf.cast(mask, tf.bool), float(-1), 0.0)

        return new_attn_mask
