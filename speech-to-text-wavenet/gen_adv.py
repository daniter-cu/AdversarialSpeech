##################################################
## CS224S, Spoken Language Processing
## Spring 2017
## Final Project
##################################################

import sugartensor as tf
from data import SpeechCorpus, voca_size
from model import *
from attacks import FastGradientMethod
import sys


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # total batch size

#
# inputs
#

# corpus input tensor
data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())

# mfcc feature of audio
inputs = tf.split(data.mfcc, tf.sg_gpus(), axis=0)
# target sentence label
labels = tf.split(data.label, tf.sg_gpus(), axis=0)

# sequence length except zero-padding
seq_len = []
for input_ in inputs:
    seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))

# parallel loss tower
'''
@tf.sg_parallel
def get_loss(opt):
    # encode audio feature
    logit = get_logit(opt.input[opt.gpu_index], voca_size=voca_size)
    # CTC loss
    return logit.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])
'''

def get_loss(input, target, seq_len):
    # encode audio feature
    logit = get_logit(input[0], voca_size=voca_size)
    # CTC loss
    return logit.sg_ctc(target=target[1], seq_len=seq_len[0])

#
# train
#
tf.sg_train(lr=0.0001, loss=get_loss(inputs, labels, seq_len),
            ep_size=data.num_batch, max_ep=50)

'''
# source: https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial_tf.py
# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
  #sess = tf.Session()
  model = lambda x: get_logit(x, voca_size=voca_size)
  fgsm = FastGradientMethod(model, sess=sess)
  fgsm_params = {'eps': 0.3}
  adv_x = fgsm.generate(inputs[0], **fgsm_params)
  print adv_x
  #sess.run([adv_x])

  # logging
  tf.sg_info('Testing finished.')
'''
