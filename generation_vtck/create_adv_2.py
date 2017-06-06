# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import librosa
from model import *
import data
from data import SpeechCorpus, voca_size, index2byte, print_index



__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 1     # batch size

#
# inputs
#

# vocabulary size
voca_size = data.voca_size

# mfcc feature of audio
x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))

# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

# encode audio feature
logit = get_logit(x, voca_size=voca_size)

# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)

# to dense tensor
y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1

data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())
mfccs = []
for mfcc_file in data.mfcc_file:
  mfcc = np.load(mfcc_file, allow_pickle=False)
  mfccs.append(mfcc.reshape((1, mfcc.shape[0], mfcc.shape[1])).transpose([0,2,1]))


# run network
with tf.Session() as sess:

    # init variables
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
    # run session
    for i in xrange(5):
      label = sess.run(y, feed_dict={x: mfccs[i]})

      # print label
      print_index(label)
