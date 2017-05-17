import sugartensor as tf
from data import SpeechCorpus, voca_size
from model import *
import numpy as np
from tqdm import tqdm
from attacks import FastGradientMethod


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

#
# inputs
#

# corpus input tensor ( with QueueRunner )
data = SpeechCorpus(batch_size=batch_size, set_name=tf.sg_arg().set)

# mfcc feature of audio
x = data.mfcc
# target sentence label
y = data.label

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

    fgsm = FastGradientMethod(get_logit, sess=sess)
    fgsm_params = {'eps': 0.3}
    adv_x = fgsm.generate(x, **fgsm_params)
    diff_x = adv_x - x
    logit_adv = get_logit_again(adv_x)
    preds_adv = get_decoded_seq(logit_adv, seq_len, y)

    logit_x = get_logit_again(x)
    preds_x = get_decoded_seq(logit_x, seq_len, y)
    #preds_adv = model(adv_x)

    # init variables
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

    # logging
    tf.sg_info('Testing started on %s set at global step[%08d].' %
               (tf.sg_arg().set.upper(), sess.run(tf.sg_global_step())))


    f = open("preds_vs_labels.tsv", "wb")
    with tf.sg_queue_context():

        # create progress bar
        iterator = tqdm(range(0, int(data.num_batch * tf.sg_arg().frac)), total=int(data.num_batch * tf.sg_arg().frac),
                        initial=0, desc='test', ncols=70, unit='b', leave=False)

        # batch loop
        loss_avg = 0.
        for _ in iterator:

            # run session
            batch_loss = None
            #batch_loss = sess.run(loss)
            adv, diff, orig_x = sess.run([adv_x, diff_x, x])
            preds, target, predsx = sess.run([preds_adv, y, preds_x])
            preds = tf.sparse_tensor_to_dense(preds, default_value=-1).eval()
            predsx = tf.sparse_tensor_to_dense(predsx, default_value=-1).eval()

            #for p, t in zip(preds, target):
            for p, t in zip(preds, predsx):
                p = [ch for ch in p if ch != -1]
                #t = [ch for ch in t if ch != 0]
                t = [ch for ch in t if ch != -1] 
                if p != t: correct = "DIFF"
                else: correct = "SAME"
                f.write("%s\t%s\t%s\n" % (correct, ' '.join(map(str,p)), ' '.join(map(str, t))))

            target_filename = "orig_x.npy"
            np.save(target_filename, orig_x, allow_pickle=False)

            target_filename = "adv_x.npy"
            np.save(target_filename, adv, allow_pickle=False)

            # loss history update
            if batch_loss is not None and \
                    not np.isnan(batch_loss.all()) and not np.isinf(batch_loss.all()):
                loss_avg += np.mean(batch_loss)

        # final average
        loss_avg /= data.num_batch * tf.sg_arg().frac

    # logging
    tf.sg_info('Testing finished on %s.(CTC loss=%f)' % (tf.sg_arg().set.upper(), loss_avg))
