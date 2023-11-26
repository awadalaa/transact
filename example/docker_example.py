import tensorflow as tf
from transact import TensorflowTransAct, TransActConfig

num_actions = 5
action_vocab = list(range(0, num_actions))
full_seq_len = 10
test_batch_size = 8
action_emb_dim = 32
item_emb_dim = 32
time_window_ms = 1000 * 60 * 60 * 1  # 1 hr
latest_n_emb = 10

# Generate random tensors in TensorFlow as input
action_type_seq = tf.random.uniform(
    shape=(test_batch_size, full_seq_len), minval=0, maxval=num_actions, dtype=tf.int32
)
item_embedding_seq = tf.random.uniform(
    shape=(test_batch_size, full_seq_len, item_emb_dim), dtype=tf.float32
)
action_time_seq = tf.random.uniform(
    shape=(test_batch_size, full_seq_len), minval=0, maxval=num_actions, dtype=tf.int32
)
request_time = tf.random.uniform(
    shape=(test_batch_size,), minval=500000, maxval=1000000, dtype=tf.int32
)
item_embedding = tf.random.uniform(
    shape=(test_batch_size, item_emb_dim), dtype=tf.float32
)
input_features = (
    action_type_seq,
    item_embedding_seq,
    action_time_seq,
    request_time,
    item_embedding,
)

# Initialize the transact module
transact_config = TransActConfig(
    action_vocab=action_vocab,
    action_emb_dim=action_emb_dim,
    item_emb_dim=item_emb_dim,
    time_window_ms=time_window_ms,
    latest_n_emb=latest_n_emb,
    seq_len=full_seq_len,
)
model = TensorflowTransAct(transact_config)

user_embedding = model(*input_features)
print(user_embedding)
