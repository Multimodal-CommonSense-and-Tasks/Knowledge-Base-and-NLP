import sys
import time

import tensorflow as tf
while not tf.io.gfile.exists(sys.argv[1]):
    time.sleep(30)
