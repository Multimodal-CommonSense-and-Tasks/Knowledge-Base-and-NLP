from dataclasses import dataclass, field
import tensorflow as tf
from util.io_utils import get_filename
from util.string_utils import unparse_args_to_cmd
from transformers import HfArgumentParser
import os


@dataclass
class ParseArgs:
    short_input_files_pattern: str = field(default="/dev/shm/pretraining_data_ko_wiki_scopa_realbert_128_whole_True_binary_fixed_wiqueen/*")
    long_input_files_pattern: str = field(default="/mnt/disks/hdd/pretraining_data_en_scopa_realbert_128_whole_True_binary_fixed_wiqueen_en/wiki_data*")
    ratio: float = field(default=1)
    output_folder_name: str = field(default='/mnt/disks/hdd/128_binary_wiqueen_ko_en_ratio10')

    max_predictions_per_seq: int = field(default=20)
    max_seq_length: int = field(default=128)

    use_full_input1: bool = field(default=True)


def write_example(writer, data):
    data = _decode_record(data, name_to_features)
    tf_example = tf.train.Example(features=tf.train.Features(feature=data))
    writer.write(tf_example.SerializeToString())


def parse_create_args():
    import sys
    parser = HfArgumentParser(ParseArgs)
    parsed_args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    sys.argv[1:] = remaining
    assert isinstance(parsed_args, ParseArgs)
    return parsed_args


FLAGS = parse_create_args()

max_seq_length = FLAGS.max_seq_length
max_predictions_per_seq = FLAGS.max_predictions_per_seq
"""The actual input function."""

name_to_features = {
    "input_ids":
        tf.io.FixedLenFeature([max_seq_length], tf.int64),
    "input_mask":
        tf.io.FixedLenFeature([max_seq_length], tf.int64),
    "segment_ids":
        tf.io.FixedLenFeature([max_seq_length], tf.int64),
    "masked_lm_positions":
        tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
    "masked_lm_ids":
        tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
    "masked_lm_weights":
        tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
    # "next_sentence_labels":
    #     tf.io.FixedLenFeature([1], tf.int64),
}

def get_dataset(input_files):
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_files)
    return d


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_string_feature(values):
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))
    return feature


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            example[name] = create_int_feature(t.numpy())
        elif t.dtype == tf.float32:
            example[name] = create_float_feature(t.numpy())
        elif t.dtype == tf.string:
            example[name] = create_string_feature(t.numpy())

    return example


def concat_ko_to_en(en_input_files, ko_wiki_dataset, output_folder, ratio):
    global example_count
    for file_i, en_input_file in enumerate(en_input_files):
        dataset = get_dataset([en_input_file])
        new_file_name = os.path.join(output_folder, get_filename(en_input_file))

        with tf.io.TFRecordWriter(new_file_name) as writer:
            for i, data in enumerate(dataset):
                if i % log_i == 0 or debug:
                    print(f"processing {i}th of {file_i}th file", flush=True)
                write_example(writer, data)
                example_count += 1

                if ratio < 1:
                    reversed_ratio = round(1 / ratio)
                    for _ in range(reversed_ratio):
                        write_example(writer, next(ko_wiki_dataset))
                    example_count = 0

                elif example_count == ratio:
                    write_example(writer, next(ko_wiki_dataset))
                    example_count = 0


if __name__ == '__main__':
    # parser = HfArgumentParser(ParseArgs)
    # print(unparse_args_to_cmd(args=parser.parse_args()))
    os.makedirs(FLAGS.output_folder_name, exist_ok=True)

    ko_input_files = []
    for input_pattern in FLAGS.short_input_files_pattern.split(","):
        ko_input_files.extend(tf.io.gfile.glob(input_pattern))

    en_wiki_input_files = []
    for input_pattern in FLAGS.long_input_files_pattern.split(","):
        en_wiki_input_files.extend(tf.io.gfile.glob(input_pattern))

    if not FLAGS.use_full_input1:
        raise NotImplementedError
    else:
        ko_wiki_dataset = get_dataset(ko_input_files)
        ko_wiki_dataset = iter(ko_wiki_dataset.repeat())

        example_count = 0
        log_i = 10000
        debug = False
        concat_ko_to_en(en_wiki_input_files, ko_wiki_dataset, FLAGS.output_folder_name, FLAGS.ratio)
