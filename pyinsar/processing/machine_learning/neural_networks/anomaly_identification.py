# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Author: Guillaume Rongier
# This software has been created in projects supported by the US National
# Science Foundation and NASA (PI: Pankratius)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import math
import numpy as np

import tensorflow as tf

def conv2d(in_layer,
           out_number_filters,
           filter_shape,
           strides = (1, 1),
           padding_mode = None,
           conv_padding = "same",
           kernel_initializer_stddev = 0.02,
           kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.1),
           normalization = tf.contrib.layers.instance_norm,
           activation = tf.nn.relu,
           name = "conv2d"):
    '''
    Define a 2D convolutional layer for a neural network
    '''
    with tf.variable_scope(name):
    
        out_layer = in_layer
        if padding_mode is not None:
            pad_shape = (int((filter_shape[0] - 1)/2),
                         int((filter_shape[1] - 1)/2))
            out_layer = tf.pad(in_layer,
                               [(0, 0), pad_shape, pad_shape, (0, 0)],
                               mode = padding_mode)

        out_layer = tf.layers.conv2d(out_layer,
                                     out_number_filters,
                                     filter_shape,
                                     strides = strides,
                                     padding = conv_padding,
                                     kernel_initializer = tf.truncated_normal_initializer(stddev = kernel_initializer_stddev),
                                     kernel_regularizer = kernel_regularizer)

        if normalization is not None:
            out_layer = normalization(out_layer)
        if activation is not None:
            out_layer = activation(out_layer)

        return out_layer

def conv2d_resize(in_layer,
                  out_number_filters,
                  filter_shape,
                  downscaling_factor,
                  out_shape = None,
                  resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                  kernel_initializer_stddev = 0.02,
                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.1),
                  normalization = tf.contrib.layers.instance_norm,
                  activation = tf.nn.relu,
                  name = "conv2d_resize"):
    '''
    Define a 2D resize layer for a neural network
    '''
    with tf.variable_scope(name):

        if out_shape is None:
            in_shape = tf.shape(in_layer)
            out_shape = tf.multiply(downscaling_factor, in_shape[1:3])
        out_layer = tf.image.resize_images(in_layer,
                                           out_shape,
                                           method = resize_method)
        
        out_layer = conv2d(out_layer,
                           out_number_filters,
                           filter_shape,
                           strides = (1, 1),
                           padding_mode = "REFLECT",
                           conv_padding = "valid",
                           kernel_initializer_stddev = kernel_initializer_stddev,
                           kernel_regularizer = kernel_regularizer,
                           normalization = normalization,
                           activation = activation)

        return out_layer
    
class EncoderDecoder(object):
    '''
    Base model for a convolutional autoencoder
    '''
    def __init__(self,
                 number_filters_first_layer = 64,
                 filter_shape = (3, 3),
                 scaling_factor = (2, 2),
                 number_scaling = 3,
                 resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 normalization = tf.contrib.layers.instance_norm,
                 regularization_scale = 0.1,
                 number_channels = 1):
        '''
        Initialize a model
        '''
        self.number_filters_first_layer = number_filters_first_layer
        self.filter_shape = filter_shape
        self.scaling_factor = scaling_factor
        self.number_scaling = number_scaling
        self.resize_method = resize_method
        self.normalization = normalization
        self.regularization_scale = regularization_scale
        self.number_channels = number_channels
        
        self.hidden_shapes = []

    def __call__(self, in_layer, reuse = tf.AUTO_REUSE, name = "generator"):
        '''
        Create the model
        '''
        with tf.variable_scope(name, reuse = reuse):
            
            out_layer = self.encode(in_layer)
            out_layer = self.decode(out_layer)
            
            return out_layer

    def encode(self, in_layer, reuse = tf.AUTO_REUSE, name = "encode"):
        '''
        Encode an image
        '''
        with tf.variable_scope(name, reuse = reuse):

            number_filters_factor = 2
            number_filters = self.number_filters_first_layer
            out_layer = conv2d(in_layer,
                               number_filters,
                               self.filter_shape,
                               strides = (1, 1),
                               padding_mode = None,
                               conv_padding = "same",
                               kernel_initializer_stddev = 0.02,
                               kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = self.regularization_scale),
                               normalization = self.normalization,
                               activation = tf.nn.leaky_relu,
                               name = "conv1")
            for i in range(self.number_scaling):
                self.hidden_shapes.append(tf.shape(out_layer)[1:3])
                number_filters /= number_filters_factor
                out_layer = conv2d(out_layer,
                                   number_filters,
                                   self.filter_shape,
                                   strides = self.scaling_factor,
                                   padding_mode = None,
                                   conv_padding = "same",
                                   kernel_initializer_stddev = 0.02,
                                   kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = self.regularization_scale),
                                   normalization = self.normalization,
                                   activation = tf.nn.leaky_relu,
                                   name = "conv" + str(i + 2))
    
            return out_layer
        
    def decode(self, in_layer, reuse = tf.AUTO_REUSE, name = "decode"):
        '''
        Decode an image
        '''
        with tf.variable_scope(name, reuse = reuse):
            
            number_filters_factor = 2
            number_filters = in_layer.get_shape().as_list()[3]
            out_layer = in_layer
            for i in range(self.number_scaling):
                number_filters *= number_filters_factor
                out_layer = conv2d_resize(out_layer,
                                          number_filters,
                                          self.filter_shape,
                                          self.scaling_factor,
                                          out_shape = self.hidden_shapes[-i - 1],
                                          resize_method = self.resize_method,
                                          kernel_initializer_stddev = 0.02,
                                          kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = self.regularization_scale),
#                                           normalization = tf.contrib.layers.instance_norm,
                                          activation = tf.nn.leaky_relu,
                                          name = "deconv" + str(i + 1))
            out_layer = conv2d(out_layer,
                               self.number_channels,
                               self.filter_shape,
                               strides = (1, 1),
                               padding_mode = None,
                               conv_padding = "same",
                               kernel_initializer_stddev = 0.02,
                               kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = self.regularization_scale),
                               normalization = self.normalization,
                               activation = None,
                               name = "conv1")

            return out_layer

class AutoencoderModel(object):
    '''
    Model for a convolutional autoencoder
    '''
    def __init__(self,
                 checkpoint_folder_path,
                 number_filters = 64,
                 filter_shape = (3, 3),
                 number_scaling = 3,
                 resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 normalization = tf.contrib.layers.instance_norm,
                 regularization_scale = 0.1,
                 number_channels = 1,
                 beta1 = 0.5,
                 dtype = tf.float32,
                 seed = 100):
        '''
        Initialize a model
        '''
        os.makedirs(checkpoint_folder_path, exist_ok = True)
        
        self.checkpoint_folder_path = checkpoint_folder_path
        self.number_filters = number_filters
        self.filter_shape = filter_shape
        self.number_scaling = number_scaling
        self.resize_method = resize_method
        self.normalization = normalization
        self.regularization_scale = regularization_scale
        self.number_channels = number_channels
        self.beta1 = beta1
        self.dtype = dtype
        self.seed = seed
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            tf.set_random_seed(self.seed)

            self.define_inputs()
            self.define_model()
            self.define_losses()
            self.define_optimizers()
            self.define_summary_variables()
            
            self.saver = tf.train.Saver()
        
    def define_inputs(self):
        '''
        Define the inputs
        '''
        with tf.variable_scope("inputs") as scope:

            self.input = tf.placeholder(self.dtype,
                                        [None, None, None, self.number_channels],
                                        name = "input")
            self.standardized_input = tf.identity(self.input)
            if self.normalization is not None:
                self.standardized_input = self.normalization(self.input)

            self.global_step = tf.Variable(0, name = "global_step", trainable = False)
            self.learning_rate = tf.placeholder(tf.float32, shape = [], name = "learning_rate")
            
    def define_model(self):
        '''
        Define the model
        '''
        with tf.variable_scope("model") as scope:
            
            encoder_decoder = EncoderDecoder(number_filters_first_layer = self.number_filters,
                                             filter_shape = self.filter_shape,
                                             number_scaling = self.number_scaling,
                                             resize_method = self.resize_method,
                                             normalization = self.normalization,
                                             regularization_scale = self.regularization_scale,
                                             number_channels = self.number_channels)

            self.encoded_input = encoder_decoder.encode(self.standardized_input, name = "encoded_input")
            self.decoded_input = encoder_decoder.decode(self.encoded_input, name = "decoded_input")
            
    def define_losses(self):
        '''
        Define the loss for training
        '''
        self.loss = tf.reduce_mean(tf.abs(self.standardized_input - self.decoded_input))
        self.loss += tf.losses.get_regularization_loss()

    def define_optimizers(self):
        '''
        Define the optimizer for training
        '''
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1)
        
        self.model_vars = tf.trainable_variables()
        variables = [var for var in self.model_vars if "encoder" or "decoder" in var.name]

        self.optimizer = optimizer.minimize(self.loss, var_list = variables)
        
    def define_summary_variables(self):
        '''
        Define summary variables for tensorboard
        '''
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        
    def save(self, session, global_step, name = "autoencoder"):
        '''
        Save the model
        '''
        self.saver.save(session,
                        os.path.join(self.checkpoint_folder_path, name),
                        global_step = global_step)
        
    def restore(self, session):
        '''
        Load a previously saved model
        '''
        latest_checkpoint_name = tf.train.latest_checkpoint(self.checkpoint_folder_path)
        if latest_checkpoint_name is not None:
            self.saver.restore(session, latest_checkpoint_name)
        
class AutoencoderTrainer(object):
    '''
    Trainer for an autoencoder
    '''
    def __init__(self, model):
        '''
        Initialize the trainer
        '''
        self.model = model
        self.training_errors = []
            
    def compute_learning_rate(self,
                              initial_learning_rate,
                              epoch,
                              learning_transition_epoch,
                              number_epochs):
        '''
        Compute the learning rate to have a linear decrease as the epoch increases
        '''
        learning_rate = initial_learning_rate
        if (epoch >= learning_transition_epoch):
            learning_rate = initial_learning_rate \
                            - initial_learning_rate*(epoch
                                                     - learning_transition_epoch)/(number_epochs
                                                                                   - learning_transition_epoch)
                
        return learning_rate
    
    def train_batch(self,
                    session,
                    training_data_array,
                    number_batches,
                    validation_data_array,
                    validation_batch_size,
                    validation_step,
                    epoch,
                    number_epochs,
                    batch,
                    batch_size,
                    learning_rate,
                    slices,
                    writer,
                    print_step):
        '''
        Train a batch of images
        '''
        batch_min = batch*batch_size
        batch_max = min((batch + 1)*batch_size, number_batches*batch_size)

        _, loss, _, summary_str = session.run([self.model.optimizer,
                                               self.model.loss,
                                               self.model.decoded_input,
                                               self.model.loss_summary], 
                                              feed_dict={self.model.input: training_data_array[batch_min:batch_max,
                                                                                               slices[0][0]:slices[0][1],
                                                                                               slices[1][0]:slices[1][1]],
                                                         self.model.learning_rate: learning_rate})
        writer.add_summary(summary_str, epoch*number_batches/batch_size + batch)
        
        if batch%print_step == 0.:
            print("Epoch: {}/{}...".format(epoch + 1, number_epochs),
                  "Batch: {}/{}...".format(batch, number_batches),
                  "Training loss: {:.4f}".format(loss))
        
        if batch%validation_step == 0. or (batch == number_batches - 1 and epoch == number_epochs - 1):
            predictions_array = np.empty(validation_data_array.shape)
            for validation_batch in range(0, validation_data_array.shape[0], validation_batch_size):

                validation_batch_max = min(validation_batch + validation_batch_size,
                                           validation_data_array.shape[0])
                predictions_array[validation_batch:validation_batch_max] = session.run(self.model.decoded_input,
                                                                                       feed_dict={self.model.input: validation_data_array[validation_batch:validation_batch_max]})
                
            validation_error = np.nanmean(np.abs(validation_data_array[:,
                                                                       slices[0][0]:slices[0][1],
                                                                       slices[1][0]:slices[1][1]] - predictions_array[:,
                                                                                                                      slices[0][0]:slices[0][1],
                                                                                                                      slices[1][0]:slices[1][1]]))
            print("Validation error:", validation_error)
            self.training_errors.append([loss, validation_error])

    def train_epoch(self,
                    session,
                    training_data_array,
                    number_batches,
                    validation_data_array,
                    validation_batch_size,
                    validation_step,
                    initial_learning_rate,
                    epoch,
                    learning_transition_epoch,
                    number_epochs,
                    batch_size,
                    slices,
                    writer,
                    print_step):
        '''
        Train once on all the batches
        '''
        learning_rate = self.compute_learning_rate(initial_learning_rate,
                                                   epoch,
                                                   learning_transition_epoch,
                                                   number_epochs)

        shuffled_training_data_array = np.random.permutation(training_data_array)

        for batch in range(0, number_batches):
            self.train_batch(session,
                             shuffled_training_data_array,
                             number_batches,
                             validation_data_array,
                             validation_batch_size,
                             validation_step,
                             epoch,
                             number_epochs,
                             batch,
                             batch_size,
                             learning_rate,
                             slices,
                             writer,
                             print_step)

        session.run(tf.assign(self.model.global_step, epoch + 1))
            
    def train(self,
              training_data_array,
              validation_data_array,
              number_epochs = 200,
              batch_size = 32,
              validation_batch_size = 1,
              initial_learning_rate = 0.0002,
              learning_transition_epoch = 100,
              slices = [[0, None], [0, None]],
              do_restore = False,
              summary_file_path = './summary',
              print_step = 1,
              validation_step = 5000,
              use_gpu = True):
        '''
        Train the autoencoder
        '''
        assert len(training_data_array.shape) == 4, "The data array must be 4D"
    
        tf.reset_default_graph()
        np.random.seed(self.model.seed)
        
        with self.model.graph.as_default():
        
            init = tf.global_variables_initializer()
            number_batches = int(training_data_array.shape[0]/batch_size)
            config = None
            if use_gpu == True:
                config = tf.ConfigProto(allow_soft_placement = True)
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count = {'GPU': 0})
            with tf.Session(config = config) as session:

                session.run(init)
                if do_restore == True:
                    self.model.restore(session)

                writer = tf.summary.FileWriter(summary_file_path, session.graph)

                for epoch in range(session.run(self.model.global_step), number_epochs):
                    self.train_epoch(session,
                                     training_data_array,
                                     number_batches,
                                     validation_data_array,
                                     validation_batch_size,
                                     validation_step,
                                     initial_learning_rate,
                                     epoch,
                                     learning_transition_epoch,
                                     number_epochs,
                                     batch_size,
                                     slices,
                                     writer,
                                     print_step)
                
                self.model.save(session, epoch)
            
class AutoencoderPredictor(object):
    '''
    Predictor for a convolutional autoencoder
    '''
    def __init__(self, model):
        '''
        Initialize the predictor
        '''
        self.model = model
            
    def predict(self, data_array, seed = 100):
        '''
        Predict as a single batch
        '''
        tf.reset_default_graph()
        
        with self.model.graph.as_default():
            
            config = tf.ConfigProto(device_count = {'GPU': 0})
            with tf.Session(config = config) as session:

                self.model.restore(session)

                return session.run(self.model.decoded_input,
                                   feed_dict={self.model.input: data_array})
        
    def batch_predict(self, data_array, batch_size = 1, encoded = False, shape = None, seed = 100):
        '''
        Predict using several batches
        '''
        tf.reset_default_graph()
        
        with self.model.graph.as_default():
            
            output = self.model.decoded_input
            output_shape = data_array.shape
            if encoded == True:
                output = self.model.encoded_input
                output_shape = shape
            
            config = tf.ConfigProto(device_count = {'GPU': 0})
            with tf.Session(config = config) as session:

                self.model.restore(session)
                
                predictions_array = np.empty(output_shape)
                for batch in range(0, data_array.shape[0], batch_size):
                    print("Batch: {}/{}...".format(batch, data_array.shape[0]))
                    
                    batch_max = min(batch + batch_size, data_array.shape[0])
                    predictions_array[batch:batch_max] = session.run(output,
                                                                     feed_dict={self.model.input: data_array[batch:batch_max]})
                    
                return predictions_array