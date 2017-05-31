#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range

from utils import *
import pdb
from time import gmtime, strftime

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    context_size = 0
    num_mfcc_features = 13
    num_final_features = num_mfcc_features * (2 * context_size + 1)

    batch_size = 1	
    num_classes = 12 # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12
    num_hidden = 128

    num_epochs = 50
    l2_lambda = 0.0000001
    lr = 5e-4#1e-2#1e-4

class CTCModel():
    """
    Implements a recursive neural network with a single hidden layer attached to CTC loss.
    This network will predict a sequence of TIDIGITS (e.g. z1039) for a given audio wav file.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph:

        inputs_placeholder: Input placeholder tensor of shape (None, None, num_final_features), type tf.float32
        targets_placeholder: Sparse placeholder, type tf.int32. You don't need to specify shape dimension.
        seq_lens_placeholder: Sequence length placeholder tensor of shape (None), type tf.int32

        TODO: Add these placeholders to self as the instance variables
            self.inputs_placeholder
            self.targets_placeholder
            self.seq_lens_placeholder

        HINTS:
            - Use tf.sparse_placeholder(tf.int32) for targets_placeholder. This is required by TF's ctc_loss op. 
            - Inputs is of shape [batch_size, max_timesteps, num_final_features], but we allow flexible sizes for
              batch_size and max_timesteps (hence the shape definition as [None, None, num_final_features]. 

        (Don't change the variable names)
        """
        inputs_placeholder = None
        targets_placeholder = None
        seq_lens_placeholder = None

        ### YOUR CODE HERE (~3 lines)
        inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.num_final_features))
        targets_placeholder = tf.sparse_placeholder(tf.int32)
        seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))
        ### END YOUR CODE

        self.inputs_placeholder = inputs_placeholder
        self.targets_placeholder = targets_placeholder
        self.seq_lens_placeholder = seq_lens_placeholder


    def create_feed_dict(self, inputs_batch, targets_batch, seq_lens_batch):
        """Creates the feed_dict for the digit recognizer.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        Args:
            inputs_batch:  A batch of input data.
            targets_batch: A batch of targets data.
            seq_lens_batch: A batch of seq_lens data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """        
        feed_dict = {} 

        ### YOUR CODE HERE (~3-4 lines)
        feed_dict[self.inputs_placeholder] = inputs_batch
        feed_dict[self.targets_placeholder] = targets_batch
        feed_dict[self.seq_lens_placeholder] = seq_lens_batch
        ### END YOUR CODE

        return feed_dict

    def add_prediction_op(self):
        """Applies a GRU RNN over the input data, then an affine layer projection. Steps to complete 
        in this function: 

        - Roll over inputs_placeholder with GRUCell, producing a Tensor of shape [batch_s, max_timestep,
          num_hidden]. 
        - Apply a W * f + b transformation over the data, where f is each hidden layer feature. This 
          should produce a Tensor of shape [batch_s, max_timesteps, num_classes]. Set this result to 
          "logits". 

        Remember:
            * Use the xavier initialization for matrices (W, but not b).
            * W should be shape [num_hidden, num_classes]. num_classes for our dataset is 12
            * tf.contrib.rnn.GRUCell, tf.contrib.rnn.MultiRNNCell and tf.nn.dynamic_rnn are of interest
        """

        logits = None 

        ### YOUR CODE HERE (~10-15 lines)
        with tf.variable_scope("prediction_op") as scope:
            self.noise = tf.get_variable("noise", shape=(1,53, Config.num_final_features), initializer=tf.zeros_initializer())
            #self.noise = tf.Variable(tf.zeros([1 ,53, Config.num_final_features]))
            perturbed_input = self.inputs_placeholder + self.noise
            W = tf.get_variable("W", [Config.num_hidden, Config.num_classes], \
                                 initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", [Config.num_classes])
            cell = tf.contrib.rnn.GRUCell(Config.num_hidden)
            cell = tf.contrib.rnn.MultiRNNCell([cell])
            output, state = tf.nn.dynamic_rnn(cell, perturbed_input, \
                          dtype=tf.float32)

            logits = tf.einsum('ijk,kl->ijl', output, W) + b
        ### END YOUR CODE

        self.logits = logits


    def add_loss_op(self):
        """Adds Ops for the loss function to the computational graph. 

        - Use tf.nn.ctc_loss to calculate the CTC loss for each example in the batch. You'll need self.logits,
          self.targets_placeholder, self.seq_lens_placeholder for this. Set variable ctc_loss to
          the output of tf.nn.ctc_loss
        - You will need to first tf.transpose the data so that self.logits is shaped [max_timesteps, batch_s, 
          num_classes]. 
        - Configure tf.nn.ctc_loss so that identical consecutive labels are allowed
        - Compute L2 regularization cost for all trainable variables. Use tf.nn.l2_loss(var). 

        """
        ctc_loss = []
        l2_cost = 0.0

        ### YOUR CODE HERE (~6-8 lines)
        logits = tf.transpose(self.logits, [1, 0, 2])
        ctc_loss = tf.nn.ctc_loss(\
                    self.targets_placeholder, \
                    logits, \
                    self.seq_lens_placeholder, \
                    ctc_merge_repeated=False, \
                    preprocess_collapse_repeated=False)
        l2_cost = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        ### END YOUR CODE

        # Remove inf cost training examples (no path found, yet)
        loss_without_invalid_paths = tf.boolean_mask(ctc_loss, tf.less(ctc_loss, tf.constant(10000.)))
        self.num_valid_examples = tf.cast(tf.shape(loss_without_invalid_paths)[0], tf.int32)
        cost = tf.reduce_mean(loss_without_invalid_paths) 

        self.loss = Config.l2_lambda * l2_cost + cost               

    def add_training_op(self):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables. The Op returned by this
        function is what must be passed to the `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model. Call optimizer.minimize() on self.loss. 

        """
        optimizer = None 

        ### YOUR CODE HERE (~1-2 lines)
        opt = tf.train.AdamOptimizer(learning_rate=Config.lr)
        optimizer = opt.minimize(self.loss, var_list=(self.noise,))
        ### END YOUR CODE
        
        self.optimizer = optimizer

    def add_decoder_and_wer_op(self):
        """Setup the decoder and add the word error rate calculations here. 

        Tip: You will find tf.nn.ctc_beam_search_decoder and tf.edit_distance methods useful here. 
        Also, report the mean WER over the batch in variable wer

        """        
        decoded_sequence = None 
        wer = None 

        ### YOUR CODE HERE (~2-3 lines)
        logits = tf.transpose(self.logits, [1, 0, 2])
        decoded_sequence, _ = tf.nn.ctc_beam_search_decoder(logits, \
                                                            self.seq_lens_placeholder)
        decoded_sequence = decoded_sequence[0] 
        wer = tf.reduce_mean(tf.edit_distance(tf.cast(decoded_sequence, tf.int32), \
                               self.targets_placeholder))
        ### END YOUR CODE

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("wer", wer)

        self.decoded_sequence = decoded_sequence
        self.wer = wer

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()


    # This actually builds the computational graph 
    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()       
        self.add_decoder_and_wer_op()
        self.add_summary_op()
        

    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch, train=True):
        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        batch_cost, wer, batch_num_valid_ex, summary = session.run([self.loss, self.wer, self.num_valid_examples, self.merged_summary_op], feed)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0
        if train:
            _ = session.run([self.optimizer], feed)

        return batch_cost, wer, summary

    def train_adversarial(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        new_loss, _, noise = session.run([self.loss, self.optimizer, self.noise], feed)
        return new_loss, noise

    def print_results(self, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        compare_predicted_to_true(train_first_batch_preds, train_targets_batch)  

    def get_pred(self, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        return (train_first_batch_preds[1], train_targets_batch[1])       

    def __init__(self):
        self.build()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', nargs='?', default='./data/hw3_train.dat', type=str, help="Give path to training data - this should not need to be changed if you are running from the assignment directory")
    parser.add_argument('--val_path', nargs='?', default='./data/hw3_val.dat', type=str, help="Give path to val data - this should not need to be changed if you are running from the assignment directory")
    parser.add_argument('--save_every', nargs='?', default=None, type=int, help="Save model every x iterations. Default is not saving at all.")
    parser.add_argument('--print_every', nargs='?', default=10, type=int, help="Print some training and val examples (true and predicted sequences) every x iterations. Default is 10")
    parser.add_argument('--save_to_file', nargs='?', default='saved_models/saved_model_epoch', type=str, help="Provide filename prefix for saving intermediate models")
    parser.add_argument('--load_from_file', nargs='?', default='saved_models/saved_model_epoch-50', type=str, help="Provide filename to load saved model")
    args = parser.parse_args()

    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    train_dataset = load_dataset(args.train_path)
    
    val_dataset = load_dataset(args.val_path)

    train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = make_batches(train_dataset, batch_size=Config.batch_size)
    val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = make_batches(train_dataset, batch_size=len(val_dataset[0]))

    def pad_all_batches(batch_feature_array):
    	for batch_num in range(len(batch_feature_array)):
    		batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
    	return batch_feature_array

    train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    num_examples = np.sum([batch.shape[0] for batch in train_feature_minibatches])
    num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))
    
    with tf.Graph().as_default():
        model = CTCModel() 
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.trainable_variables()[1:])

        with tf.Session() as session:
            # Initializate the weights and biases
            session.run(init)
            if args.load_from_file is not None:
            	new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
                new_saver.restore(session, args.load_from_file)
            
            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)

            global_start = time.time()

            step_ii = 0

            total_train_cost = total_train_wer = 0
            start = time.time()

            batch_ii = 15
            tmp = train_labels_minibatches[batch_ii]
            new_label = (tmp[0], np.array([1], dtype=np.int32), tmp[2])

            np.save("before.npy", train_feature_minibatches[batch_ii])

            for i in xrange(1000):
                new_loss, noise = model.train_adversarial(session, train_feature_minibatches[batch_ii], new_label, train_seqlens_minibatches[batch_ii])
                pred, target = model.get_pred(train_feature_minibatches[batch_ii], train_labels_minibatches[batch_ii], train_seqlens_minibatches[batch_ii])
                if i % 100 == 0:
                    print("\n")
                    print("Iteration: ",i)
                    print("Loss: ", new_loss)
                    print("Pred: ", pred)
                    print("Target: ", target)
                if len(pred) == 1 and pred[0] == 1:
                    print("Loss: ", new_loss)
                    print("Successfully fooled network!")
                    break
            print(np.max(noise))
            print(np.min(noise))
            print(np.mean(noise))
            print(np.max(train_feature_minibatches[batch_ii]))
            print(np.min(train_feature_minibatches[batch_ii]))
            print(np.mean(train_feature_minibatches[batch_ii]))

            np.save("after.npy", train_feature_minibatches[batch_ii] + noise)

            

