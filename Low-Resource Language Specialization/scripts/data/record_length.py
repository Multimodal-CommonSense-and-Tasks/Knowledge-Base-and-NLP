import sys
import tensorflow as tf

record_name = sys.argv[1]

print(sum(1 for _ in tf.python_io.tf_record_iterator(record_name)))
