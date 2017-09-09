import tensorflow as tf
flags = tf.flags
logging = tf.logging

flags.DEFINE_string('para_name_1', 'default_val', 'description')
flags.DEFINE_bool('para_name_val', 'default_val', 'description')

FLAGS = flags.FLAGS


def main():
    FLAGS.para_name_1


if __name__ == '__main__':
    tf.app.run()
