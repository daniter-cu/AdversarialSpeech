import sugartensor as tf
from data import SpeechCorpus, voca_size, index2str
from model import *
import numpy as np
from tqdm import tqdm
from attacks import FastGradientMethod
import pickle

##################################################
## Edited by Jade Huang
##################################################

__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)

# command line argument for set_name
tf.sg_arg_def(set=('valid', "'train', 'valid', or 'test'.  The default is 'valid'"))
tf.sg_arg_def(frac=(1.0, "test fraction ratio to whole data set. The default is 1.0(=whole set)"))


#
# hyper parameters
#

# batch size
batch_size = 16

num_iters = 15
print "num_iters: %s" % num_iters

#
# inputs
#

# corpus input tensor ( with QueueRunner )
print 'building corpus'
data = SpeechCorpus(batch_size=batch_size, set_name=tf.sg_arg().set)

# mfcc feature of audio
x = data.mfcc
# target sentence label
y = data.label

filenames_t = data.filenames

# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

#
# Testing Graph
#

# encode audio feature
#logit = get_logit(x, voca_size=voca_size)

# CTC loss
#loss = logit.sg_ctc(target=y, seq_len=seq_len)

#
# run network
#

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # generate adversarial examples
    fgsm = FastGradientMethod(get_logit, sess=sess)
    fgsm_params = {'eps': 0.3}

    # run adversarial examples through network
    adv_x = fgsm.generate(x, **fgsm_params)
    logit_adv = get_logit_again(adv_x)
    preds_adv = get_decoded_seq(logit_adv, seq_len, y)

    def get_adv(x, num_iters):
      fgsm = FastGradientMethod(get_logit_again, sess=sess)
      advx = fgsm.generate(x, **fgsm_params)
      
      for _ in xrange(num_iters - 1):
        advx = fgsm.generate(advx, **fgsm_params)
      return advx

    def get_logit_adv(x, num_iters):
      return get_logit_again(get_adv(x, num_iters))
 
    def get_preds_adv(x, num_iters):
      return get_decoded_seq(get_logit_adv(x, num_iters), seq_len, y)

    # output on normal inputs
    logit_x = get_logit_again(x)
    preds_x = get_decoded_seq(logit_x, seq_len, y)

    # init variables
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

    # logging
    tf.sg_info('Testing started on %s set at global step[%08d].' %
               (tf.sg_arg().set.upper(), sess.run(tf.sg_global_step())))


    f = open("vctk/preds_vs_labels_iters" + str(num_iters) + ".tsv", "wb")
    #f = open("one_word/preds_vs_labels_iters" + str(num_iters) + ".tsv", "wb")
    f.write("filename\tsame_diff\tpred_on_orig\tpred_on_adv\ttarget\tnum_pred_on_orig\tnum_pred_on_adv\tnum_target\n")
    #orig_x_f = open("one_word/orig_x_iters" + str(num_iters) + ".npy", "wb")
    orig_x_f = open("vctk/orig_x_iters" + str(num_iters) + ".npy", "wb")
    #adv_x_f = open("one_word/adv_x_iters" + str(num_iters) + ".npy", "wb")
    adv_x_f = open("vctk/adv_x_iters" + str(num_iters) + ".npy", "wb")
    orig_xs = []
    adv_xs = []

    with tf.sg_queue_context():

        # create progress bar
        iterator = tqdm(range(0, int(data.num_batch * tf.sg_arg().frac)), total=int(data.num_batch * tf.sg_arg().frac),
                        initial=0, desc='test', ncols=70, unit='b', leave=False)

        # batch loop
        loss_avg = 0.

        for _ in iterator:

          # get original inputs once
          orig_x, target, predsx, filenames, adv, preds= sess.run([x, y, preds_x, filenames_t, \
            get_adv(x, num_iters), get_preds_adv(x, num_iters)])
          orig_xs.append(orig_x)
          adv_xs.append(adv)

          predsx = tf.sparse_tensor_to_dense(predsx, default_value=-1).eval()
          preds = tf.sparse_tensor_to_dense(preds, default_value=-1).eval()

          #batch_loss = None

          for p, px, t, filename in zip(preds, predsx, target, filenames):
              p = [(int(ch) + 1) for ch in p if ch != -1]
              str_p = index2str(p)
              
              t = [ch for ch in t if ch != 0]
              str_t = index2str(t)

              px = [(int(ch) + 1) for ch in px if ch != -1] 
              str_px = index2str(px)
              
              if px != p: correct = "DIFF"
              else: correct = "SAME"

              f.write("%s\n" % '\t'.join([filename, correct, ''.join(map(str, str_px)), ''.join(map(str,str_p)), ''.join(map(str, str_t)), ' '.join(map(str, px)), ' '.join(map(str, p)), ' '.join(map(str, t))]))

              # loss history update
              #if batch_loss is not None and \
              #        not np.isnan(batch_loss.all()) and not np.isinf(batch_loss.all()):
              #    loss_avg += np.mean(batch_loss)

            # final average
            #loss_avg /= data.num_batch * tf.sg_arg().frac

        # logging
        tf.sg_info('Testing finished on %s.(CTC loss=%f)' % (tf.sg_arg().set.upper(), loss_avg))

        # save arrays to file
        pickle.dump(orig_xs, orig_x_f)
        pickle.dump(adv_xs, adv_x_f)
        #np.save(orig_x_f, orig_xs, allow_pickle=False)
        #np.save(adv_x_f, adv_xs, allow_pickle=False)
