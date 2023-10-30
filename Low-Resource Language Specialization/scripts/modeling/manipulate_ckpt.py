import sys

import tensorflow.compat.v1 as tf


class CkptChanger:
    def __init__(self, ckpt_dir):
        self._checkpoint = self._get_ckpt(ckpt_dir)
        self._name2val = {}

        self._load(self._checkpoint)

    def _get_ckpt(self, checkpoint_dir):
        """
        If input is checkpoint file path name, returns it directly.
        Else, gets the latest checkpoint file in it.
        """
        checkpoint_filename = checkpoint_dir
        if tf.gfile.IsDirectory(checkpoint_dir):
            checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
            if checkpoint_state is None:
                print('Checkpoint file not found in {}.'.format(checkpoint_dir))
                sys.exit(1)
            checkpoint_filename = checkpoint_state.model_checkpoint_path

        return checkpoint_filename

    def _load(self, ckpt_dir):
        name_shape_tuple_lists = tf.train.list_variables(ckpt_dir)
        for name, shape in name_shape_tuple_lists:
            np_tensor = tf.train.load_variable(self._checkpoint, name)
            assert tuple(shape) == tuple(np_tensor.shape)

            self._name2val[name] = np_tensor

    def save(self, ckpt_dir=None):
        if ckpt_dir is None:
            ckpt_dir = self._checkpoint

        tf.reset_default_graph()
        with tf.Session() as session:
            for name, value in self._name2val.items():
                var = tf.Variable(value, name=name)
            session.run(tf.global_variables_initializer())
            tf.train.Saver().save(session, ckpt_dir, write_meta_graph=False, write_state=False)

    def get_all_var_names(self):
        return list(self._name2val.keys())

    def get_val(self, var_name):
        return self._name2val[var_name]

    def set_val(self, var_name, np_array):
        self._name2val[var_name] = np_array

    def del_val(self, var_name):
        self._name2val.pop(var_name)
