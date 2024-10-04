import sys
import time

try:
    import tensorflow as tf
    while not tf.io.gfile.exists(sys.argv[1]):
        time.sleep(30)
except:
    import os
    while not os.path.exists(sys.argv[1]):
        time.sleep(30)
