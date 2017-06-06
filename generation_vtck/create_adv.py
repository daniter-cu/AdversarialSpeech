import sugartensor as tf
from data import SpeechCorpus, voca_size, index2byte, print_index
from model import *


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 1    # total batch size

#
# inputs
#

# corpus input tensor
data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())

# mfcc feature of audio
x = data.mfcc
# target sentence label
y = data.label

seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

# encode audio feature
logit = get_logit(x, voca_size=voca_size)

# CTC loss
loss = logit.sg_ctc(target=y, seq_len=seq_len)

decoded_sequence, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)
#decoded_sequence = decoded_sequence[0] 


#x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))
y = tf.sparse_to_dense(decoded_sequence[0].indices, decoded_sequence[0].dense_shape, decoded_sequence[0].values) + 1

#
# train
#0.0001
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
  # init variables
  tf.sg_init(sess)

  # restore parameters
  saver = tf.train.Saver()
  saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

  #epoch[41182]-step[205919]

  #tf.sg_train(lr=0, loss=get_loss(input=inputs, target=labels, seq_len=seq_len),
  #        ep_size=data.num_batch, max_ep=41182+5, sess=sess, max_keep=0, keep_interval=0, save_interval=0)
  with tf.sg_queue_context():
    for _ in xrange(5):
      out = sess.run(y)
      print_index(out)











