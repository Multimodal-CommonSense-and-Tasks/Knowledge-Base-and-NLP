import argparse

from transformers import (
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

# most of these args follow the defaults at https://huggingface.co/blog/how-to-train
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train-file", type=str, help="Training files", required=True
)
parser.add_argument("--output-dir", type=str, help="Output dir", required=True)
parser.add_argument(
    "--batch-size", type=int, help="Batch size for training", required=True
)
parser.add_argument(
    "--train-epochs", type=int, help="Number of epochs", default=20
)
# parser.add_argument(
#     "--save-steps",
#     type=int,
#     help="How often to save (in steps)",
#     required=True,
# )
parser.add_argument(
    "--vocab-size", type=int, help="Size of pretrained vocab", default=52000
)
parser.add_argument(
    "--vocab-dir", type=str, help="Path to vocab root", required=True
)
parser.add_argument("--mlm-probability", type=float, default=0.15)
parser.add_argument("--block-size", type=int, default=128)
parser.add_argument("--max-position-embeddings", type=int, default=514)
parser.add_argument("--num-attention-heads", type=int, default=12)
parser.add_argument("--num-hidden-layers", type=int, default=6)
parser.add_argument("--type-vocab-size", type=int, default=1)

args = parser.parse_args()

config = RobertaConfig(
    vocab_size=args.vocab_size,
    max_position_embeddings=args.max_position_embeddings,
    num_attention_heads=args.num_attention_heads,
    num_hidden_layers=args.num_hidden_layers,
    type_vocab_size=args.type_vocab_size,
)

# this was default 512; assuming it's position embeddings minus <s> and </s>
tokenizer = RobertaTokenizerFast.from_pretrained(
    args.vocab_dir, max_len=args.max_position_embeddings - 2
)

model = RobertaForMaskedLM(config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer, file_path=args.train_file, block_size=args.block_size
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.train_epochs,
    per_device_train_batch_size=args.batch_size,
    save_steps=1,
    save_total_limit=None,
    # let's not pass in the dev set for now for consistency with mBERT models
    evaluation_strategy="no",
    # this is from the blog but no the notebook
    learning_rate=1e-4,
    do_train=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# surely there has to be a better way than this?
steps_per_epoch = len(trainer.get_train_dataloader())
trainer.args.save_steps = steps_per_epoch
trainer.args.logging_steps = steps_per_epoch
# this loosely follows UDify, RoBERTa, and parsing-mbert
# (linear warmup for 1 epoch)
trainer.args.warmup_steps = steps_per_epoch

trainer.train()

trainer.save_model(args.output_dir)
