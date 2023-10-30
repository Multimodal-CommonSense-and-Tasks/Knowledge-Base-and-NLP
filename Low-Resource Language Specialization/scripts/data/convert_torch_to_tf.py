# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import argparse
import os

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf2
tf2.config.experimental.set_visible_devices([], 'GPU')
import torch

# from transformers import BertModel
from transformers.models.bert.modeling_bert import BertModel, BertForPreTraining


def convert_pytorch_checkpoint_to_tf(model: BertForPreTraining, ckpt_dir: str, model_name: str):

    """
    Args:
        model: BertModel Pytorch model instance to be converted
        ckpt_dir: Tensorflow model directory
        model_name: model name

    Currently supported HF models:

        - Y BertModel
        - N BertForMaskedLM
        - Y BertForPreTraining
        - N BertForMultipleChoice
        - N BertForNextSentencePrediction
        - N BertForSequenceClassification
        - N BertForQuestionAnswering
    """

    tensors_to_transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")
    tensors_to_skip = ("decoder",)

    var_map = (
        ("layer.", "layer_"),
        ("word_embeddings.weight", "word_embeddings"),
        ("position_embeddings.weight", "position_embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings"),
        (".", "/"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("weight", "kernel"),
        ("seq_relationship/kernel", "seq_relationship/output_weights"), # originally seq_relationship/kernel, but changed because of previous...
        ("seq_relationship/bias", "seq_relationship/output_bias"),
        ("predictions/bias", "predictions/output_bias"),
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    prefix = '' if hasattr(model, 'bert') else 'bert/'
    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return f"{prefix}{name}"

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            if any([x in var_name for x in tensors_to_skip]):
                print([x in var_name for x in tensors_to_skip])
                print(f"skipping {var_name}")
                continue
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print(f"Successfully created {tf_name}: {np.allclose(tf_weight, torch_tensor)}")

        saver = tf.train.Saver(tf.trainable_variables())
        from util.io_utils import get_filename
        saver.save(session, os.path.join(ckpt_dir, get_filename(model_name, remove_ext=False).replace("-", "_") + ".ckpt"))


from typing import List
def copy_key(model, model_with_real_weight, hierarchical_keys: List[str]):
    if hierarchical_keys:
        copy_key(getattr(model, hierarchical_keys[0]), getattr(model_with_real_weight, hierarchical_keys[0]),
                 hierarchical_keys[1:])
    else:
        model.data.copy_(model_with_real_weight.data)


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model name e.g. bert-base-uncased")
    parser.add_argument(
        "--cache_dir", type=str, default=None, required=False, help="Directory containing pytorch model"
    )
    parser.add_argument("--orig_model_path", type=str, help="/path/to/<pytorch-model-name>.bin")
    parser.add_argument("--tf_cache_dir", type=str, required=True, help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)

    bert_cls = BertForPreTraining
    model, loading_info = bert_cls.from_pretrained(args.model_name, output_loading_info=True)
    if args.orig_model_path and ('cls.predictions.transform.dense.weight' in loading_info['missing_keys']):
        orig_model, orig_loading_info = bert_cls.from_pretrained(args.orig_model_path, output_loading_info=True)
        assert not 'cls.predictions.transform.dense.weight' in orig_loading_info['missing_keys']
        for key in loading_info['missing_keys']:
            copy_key(model, orig_model, key.split('.'))

    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir=args.tf_cache_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()
