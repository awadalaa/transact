# TransAct [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fawadalaa%2Ftransact)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fawadalaa%2Ftransact)

![PyPI](https://img.shields.io/pypi/v/transact-tf)
[![Run Tests](https://github.com/awadalaa/transact/actions/workflows/tests.yml/badge.svg)](https://github.com/awadalaa/transact/actions/workflows/tests.yml)
[![Upload Python Package](https://github.com/awadalaa/transact/actions/workflows/python-publish.yml/badge.svg)](https://github.com/awadalaa/transact/actions/workflows/python-publish.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Super-Linter](https://github.com/awadalaa/transact/actions/workflows/linter.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)


![GitHub License](https://img.shields.io/github/license/awadalaa/transact)
[![GitHub stars](https://img.shields.io/github/stars/awadalaa/transact?style=social)](https://github.com/awadalaa/transact/stargazers)
[![GitHub followers](https://img.shields.io/github/followers/awadalaa?label=Follow&style=social)](https://github.com/awadalaa)
[![Twitter Follow](https://img.shields.io/twitter/follow/awadalaa?style=social)](https://twitter.com/intent/follow?screen_name=awadalaa)

This is NOT the official implementation by the authors of this model architecture. You can find the official pytorch [implementation here](https://github.com/pinterest/transformer_user_action). This repo is a **Tensorflow** implementation of [TransAct: Transformer-based Realtime User Action Model for
Recommendation at Pinterest](https://dl.acm.org/doi/10.1145/3580305.3599918) by Xia, Xue, et al. **TransAct** is the ranking architecture 
used by Pinterest's Homefeed to personalize and extract users' short-term preferences from their realtime activities. The paper was presented at KDD 2023.

![](https://github.com/awadalaa/transact/blob/main/media/architecture.png)

## Installation

### PyPI - Not Working Yet

New user registration on PyPI is temporarily suspended due to malicious attacks. Once admins enable, will add. Until then, skip ahead to the docker step.

Run the following to install:

```sh
pip install transact-tf
```

### Docker

To install the package using Docker run the following:

```sh
docker pull ghcr.io/awadalaa/transact:release
```

## Developing transact

To install `transact`, along with tools you need to develop and test, run the following in your virtualenv:

```sh
git clone https://github.com/awadalaa/transact.git
# or clone your own fork

cd transact
pip install -e .
```

To run rank and shape tests run any of the following:

```py
python -m transact.test_transact
pytest transact --verbose
```

## Usage

```python
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
action_type_seq = tf.random.uniform(shape=(test_batch_size, full_seq_len), minval=0, maxval=num_actions, dtype=tf.int32)
item_embedding_seq = tf.random.uniform(shape=(test_batch_size, full_seq_len, item_emb_dim), dtype=tf.float32)
action_time_seq = tf.random.uniform(shape=(test_batch_size, full_seq_len), minval=0, maxval=num_actions, dtype=tf.int32)
request_time = tf.random.uniform(shape=(test_batch_size,), minval=500000, maxval=1000000, dtype=tf.int32)
item_embedding = tf.random.uniform(shape=(test_batch_size, item_emb_dim), dtype=tf.float32)
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

```

## Run with Docker

You can also run the example script with Docker.

```sh
git clone https://github.com/awadalaa/transact.git
cd transact

docker run -it --rm \
    --mount type=bind,source="$(pwd)"/example,target=/usr/src/transact/docker_example \
    ghcr.io/awadalaa/transact:release \
    python docker_example/docker_example.py
```

## Want to Contribute 🙋‍♂️?

Awesome! If you want to contribute to this project, you're always welcome! See [Contributing Guidelines](CONTRIBUTING.md). You can also take a look at [open issues](https://github.com/awadalaa/transact/issues) for getting more information about current or upcoming tasks.

## Want to discuss? 💬

Have any questions, doubts or want to present your opinions, views? You're always welcome. You can [start discussions](https://github.com/awadalaa/transact/discussions).

## Citation

```bibtex
@article{xia2023transact,
  title={TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest},
  author={Xia, Xue and Eksombatchai, Pong and Pancha, Nikil and Badani, Dhruvil Deven and Wang, Po-Wei and Gu, Neng and Joshi, Saurabh Vishwas and Farahpour, Nazanin and Zhang, Zhiyuan and Zhai, Andrew},
  journal={arXiv preprint arXiv:2306.00248},
  year={2023}
}
```

## License

```
Copyright 2023 Alaa Awad

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
