from transformers import HfArgumentParser
from transformers.models.bert.modeling_bert import BertForPreTraining, BertModel
from dataclasses import dataclass, field
import tensorflow as tf
import torch
try:
    tf.enable_eager_execution()
except AttributeError:
    pass
tf.config.experimental.set_visible_devices([], 'GPU')
import torch.nn as nn

@dataclass
class Args:
    model: str
    save_dir: str
    ref_model: str
    init_layer: int = field(default=13, metadata={"help": "max 11"})
    reinit_pooler: bool = False
    init_std: float = 0.02

parser = HfArgumentParser(Args)
args, = parser.parse_args_into_dataclasses()
assert isinstance(args, Args)

model = BertForPreTraining.from_pretrained(args.model)
bert_model = model.bert
assert isinstance(bert_model, BertModel)

ref_model = BertForPreTraining.from_pretrained(args.ref_model)
model.cls = ref_model.cls
ref_bert_model = ref_model.bert
bert_model.pooler = ref_bert_model.pooler

def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        orig_weight = module.weight.data
        new_weight = tf.random.truncated_normal(shape=orig_weight.shape, mean=0.0, stddev=args.init_std)
        module.weight.data.copy_(torch.tensor(new_weight.numpy()))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        raise NotImplementedError
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


for i, layer in enumerate(bert_model.encoder.layer):
    if i >= args.init_layer:
        if i != args.init_layer:
            bert_model.encoder.layer[i] = ref_bert_model.encoder.layer[i]
        else:
            layer.apply(_init_weights)

model.save_pretrained(args.save_dir)