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

index = 0

lr = 1e-2

#
# inputs
#
corpus = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())
mfccs = []
for mfcc_file in corpus.mfcc_file:
  mfcc = np.load(mfcc_file, allow_pickle=False)
  mfccs.append(mfcc.reshape((1, mfcc.shape[0], mfcc.shape[1])).transpose([0,2,1]))

# vocabulary size
voca_size = data.voca_size

# mfcc feature of audio
x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))

noise = tf.get_variable("noise", shape=(batch_size, mfccs[index].shape[1], 20), initializer=tf.zeros_initializer())

perturbed_input = x + noise

# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

# encode audio feature
logit = get_logit(perturbed_input, voca_size=voca_size)

# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)

# to dense tensor
pred = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1

targ = tf.placeholder(dtype=tf.int32, shape=corpus.label.shape)#corpus.label
loss = logit.sg_ctc(target=targ, seq_len=seq_len)

opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
optimizer = opt.minimize(loss, var_list=(noise,))

# run network
with tf.Session() as sess:

    # init variables
    tf.sg_init(sess)

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + \
          tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)

    vars_to_train = [el for el in all_vars if 'noise' not in el.name]

    # restore parameters
    saver = tf.train.Saver(vars_to_train)
    #saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
    # run session
    #with tf.sg_queue_context():
    for i in xrange(100):
      if i % 10 == 0:
        print "iteration ", i
      new_loss, _, noise_out = sess.run([loss, optimizer, noise], feed_dict={x: mfccs[index], targ:corpus.daniter_label[index].reshape((1, -1))})

    label = sess.run(pred, feed_dict={x: mfccs[index]})

    # print label
    print_index(label)
