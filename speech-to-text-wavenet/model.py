import sugartensor as tf
from data import voca_size


num_blocks = 3     # dilated blocks
num_dim = 128      # latent dimension


#
# logit calculating graph using atrous convolution
#
def get_logit(x):

    # residual block
    def res_block(tensor, size, rate, block, dim=num_dim):

        with tf.sg_context(name='block_%d_%d' % (block, rate)):

            # filter convolution
            conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True, name='conv_filter')

            # gate convolution
            conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True, name='conv_gate')

            # output by gate multiplying
            out = conv_filter * conv_gate

            # final output
            out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True, name='conv_out')

            # residual and skip output
            return out + tensor, out

    # expand dimension
    with tf.sg_context(name='front'):
        z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True, name='conv_in')

    # dilated conv block loop
    skip = 0  # skip connections
    for i in range(num_blocks):
        for r in [1, 2, 4, 8, 16]:
            z, s = res_block(z, size=7, rate=r, block=i)
            skip += s

    # final logit layers
    with tf.sg_context(name='logit'):
        logit = (skip
                 .sg_conv1d(size=1, act='tanh', bn=True, name='conv_1')
                 .sg_conv1d(size=1, dim=voca_size, name='conv_2'))

    #variables_names = [v.name for v in tf.trainable_variables()]
    #print variables_names
    return logit
#
# logit calculating graph using atrous convolution
#
def get_logit_again(x):

    # residual block
    def res_block(tensor, size, rate, block, dim=num_dim):

        #with tf.sg_context(name='block_%d_%d' % (block, rate)):
        with tf.sg_context(name='block_%d_%d' % (block, rate), reuse=True):

            # filter convolution
            conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True, name='conv_filter')

            # gate convolution
            conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True, name='conv_gate')

            # output by gate multiplying
            out = conv_filter * conv_gate

            # final output
            out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True, name='conv_out')

            # residual and skip output
            return out + tensor, out

    # expand dimension
    #with tf.sg_context(name='front'):
    with tf.sg_context(name='front', reuse=True):
        z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True, name='conv_in')

    # dilated conv block loop
    skip = 0  # skip connections
    for i in range(num_blocks):
        for r in [1, 2, 4, 8, 16]:
            z, s = res_block(z, size=7, rate=r, block=i)
            skip += s

    # final logit layers
    #with tf.sg_context(name='logit'):
    with tf.sg_context(name='logit', reuse=True):
        logit = (skip
                 .sg_conv1d(size=1, act='tanh', bn=True, name='conv_1')
                 .sg_conv1d(size=1, dim=voca_size, name='conv_2'))

    #variables_names = [v.name for v in tf.trainable_variables()]
    #print variables_names
    return logit

def get_decoded_seq(logits, seq_lens, targets):
  logits = tf.transpose(logits, [1, 0, 2])
  decoded_sequence, _ = tf.nn.ctc_beam_search_decoder(logits, \
                                                            seq_lens)
  # TODO: what is the exact dimension that's returned?
  decoded_sequence = decoded_sequence[0]
 
  return decoded_sequence

