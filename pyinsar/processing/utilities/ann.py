# Standard library imports
from collections import OrderedDict
import os

# 3rd party imports
import tensorflow as tf
import numpy as np


def buildCNN(image_height, image_width, model_dir, rate=0.01, config=None):
    """
    Build a convolutional neural network

    @param image_height: Height of image in pixels
    @param image_width: Width of image in pixels
    @param model_dir: Directory to save network too
    @param rate: Learning rate
    @param config: Config to pass to tf.Session
    """
    graph = tf.Graph()

    with tf.Session(graph=graph, config=config) as session:
        training = tf.placeholder_with_default(False, shape=(), name='Training')

        with tf.variable_scope('IO'):
            data_input = tf.placeholder(tf.float32, shape=(None, image_height, image_width, 1), name = 'Input')
            output = tf.placeholder(tf.int64, shape=(None), name='Output')

        with tf.name_scope('Clean'):
            normalize = tf.map_fn(lambda x: tf.image.per_image_standardization(x), data_input, name='Normalize')

        with tf.name_scope('Convolution_Layers'):
            new_image_height = image_height
            new_image_width = image_width

            conv1 = tf.layers.conv2d(normalize, filters=40, kernel_size=[9,9],padding='valid', activation=tf.nn.relu,
                                     name = 'Convolution_1')
            new_image_height = length_after_valid_window(new_image_height, 9, 1)
            new_image_width = length_after_valid_window(new_image_width, 9, 1)

            pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2, name = 'Max_Pool_1')
            new_image_height = length_after_valid_window(new_image_height, 2, 2)
            new_image_width = length_after_valid_window(new_image_width, 2, 2)


            conv2 = tf.layers.conv2d(pool1, filters=20, kernel_size=[5,5],padding='valid', activation=tf.nn.relu,
                                     name = 'Convolution_2')
            new_image_height = length_after_valid_window(new_image_height, 5, 1)
            new_image_width = length_after_valid_window(new_image_width, 5, 1)

            pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2, name = 'Max_Pool_2')
            new_image_height = length_after_valid_window(new_image_height, 2, 2)
            new_image_width = length_after_valid_window(new_image_width, 2, 2)

            pool2_flat = tf.reshape(pool2, [-1, new_image_height*new_image_width*20], name='Reshape')

        with tf.name_scope('Fully_Connected_Layers'):
                dense1 = tf.layers.dense(inputs=pool2_flat, units=1000, name='Dense_1')
                batch_norm1 = tf.layers.batch_normalization( dense1, training=training, momentum=0.9)
                activate1 = tf.nn.relu(batch_norm1, name='Activate_1')

                dense2 = tf.layers.dense(inputs=activate1, units=100, name='Dense_2')
                batch_norm2 = tf.layers.batch_normalization(dense2, training=training, momentum=0.9)
                activate2 = tf.nn.relu(batch_norm2, name='Activate_2')

                logits = tf.layers.dense(inputs=activate2, units=2, name='Logits')

                # update_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.name_scope('Loss'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=output, name='Entropy')
            loss = tf.reduce_mean(entropy, name='Loss')

        with tf.name_scope('Train'):
            gradient = tf.train.GradientDescentOptimizer(rate)
            global_step = tf.train.get_or_create_global_step()
            minimize = gradient.minimize(loss, global_step = global_step, name='Minimize')

        with tf.name_scope('Evaluate'):
            correct_responses = tf.nn.in_top_k(logits, output, 1, name='Correct_Responses')
            evaluate = tf.reduce_mean(tf.cast(correct_responses, tf.float32), name='Evaluate')

        with tf.name_scope('Initialization'):
            initializer = tf.global_variables_initializer()



        graph.add_to_collection('fit', minimize)
        graph.add_to_collection('input', data_input)
        graph.add_to_collection('output', output)
        graph.add_to_collection('train', training)
        graph.add_to_collection('global_step', global_step)
        graph.add_to_collection('initializer', initializer)
        graph.add_to_collection('evaluate', evaluate)
        graph.add_to_collection('logits', logits)

        initializer.run()
        saver = tf.train.Saver(name='Saver')
        saver.save(session, os.path.join(model_dir,'network'))


def train(image_data, image_labels, model_dir,
          batch_size, num_epochs,  max_batches=None,
          status_line_rate = 50, target='', shuffle=True,
          config=None):
    """
    Train neural network

    @param image_data: Image data to train (shape [:,image_width, image_height])
    @param image_labels: Corresponding labels
    @param model_dir: Directory where network is stored
    @param batch_size: Batch size
    @param num_epochs: Number of epochs
    @param max_batches: Max number of patches (Typically used for testing)
    @param status_line_rate: Number of batches between outputting training information
    @param target: Unused
    @param shuffle: Whether or not to shuffle the training data
    @param config: Config to pass to tf.Session
    """

    num_batches = image_data.shape[0] // batch_size

    if max_batches != None:
        num_batches = min(max_batches, num_batches)

    model_filename = os.path.join(model_dir, 'network')
    graph, op_dict, model_checkpoint = restoreGraph(model_dir)

    train_op = op_dict['train']
    input_placeholder = op_dict['input']
    output_placeholder = op_dict['output']
    global_step = op_dict['global_step']
    fit_op =op_dict['fit']
    evaluate = op_dict['evaluate']
    saver = op_dict['saver']


    with tf.Session(graph=graph, config=config) as session:
        saver.restore(session, model_checkpoint)

        for epoch in range(num_epochs):

            if shuffle:
                image_data, image_labels = shuffleTrainingData(image_data, image_labels)

            for index in range(num_batches):
                run_id = tf.train.global_step(session, global_step)
                batch_slice = slice(index*batch_size, (index+1)*batch_size)
                train_data = image_data[batch_slice].reshape(-1,100,100,1)
                train_labels = image_labels[batch_slice]
                batch_dict = {input_placeholder : train_data, output_placeholder: train_labels}
                if train_op != None:
                    batch_dict[train_op] = True

                session.run(fit_op, feed_dict=batch_dict)

                if (run_id+1) % status_line_rate == 0:
                    accuracy = evaluate.eval(feed_dict={input_placeholder : train_data, output_placeholder: train_labels})
                    print('Batch accuracy after global step ', str(run_id).zfill(6), ": ",
                          '{:04.2f}'.format(accuracy), sep='')

            saver.save(session, model_filename, run_id)

def classify(image_data, model_dir, batch_size=2000, config=None):
    """
    Classify data

    @param image_data: Input data
    @param model_dir: Directory where network is stored
    @param batch_size: Batch size to use for classifying data
    @param config: Config to pass on to tf.Session

    @return Predicted labels for input data
    """

    graph, op_dict, model_checkpoint = restoreGraph(model_dir)

    input_placeholder = op_dict['input']
    logits = op_dict['logits']
    saver = op_dict['saver']

    with tf.Session(graph=graph, config=config) as session:
        saver = tf.train.Saver()
        saver.restore(session, model_checkpoint)

        results = []
        num_images = image_data.shape[0]
        num_batches = np.ceil(num_images / batch_size).astype('int')

        for index in range(num_batches):
            slice_index = slice(index*batch_size, min((index+1)*batch_size, num_images))
            batched_data = {input_placeholder: image_data[slice_index].reshape(-1, image_data.shape[1], image_data.shape[2], 1)}

            results.append(np.argmax(logits.eval(batched_data), axis=1))

    return np.concatenate(results)


def length_after_valid_window(length, window, stride):
    """
    Length of dimension after convolving using the padding type 'valid' or using max pooling

    @param length: Initial length
    @param window: Size of convolution window
    @param stride: Stride used
    @return New size after using convolution with 'valid' padding type or from max pooling
    """
    return np.ceil( (length - window + 1) / stride).astype('int')

def shuffleTrainingData(data, labels):
    """
    Shuffles data

    @param data: Input data
    @param labels: Input labels
    """
    index = np.arange(data.shape[0])
    np.random.shuffle(index)

    return data[index], labels[index]

def restoreGraph(model_dir):
    """
    Restore a network

    @param model_dir: Directory containing network
    @return graph, operation dictionary, and checkpoint
    """

    graph = tf.Graph()

    op_dict = OrderedDict()

    with graph.as_default():
        model_checkpoint = tf.train.latest_checkpoint(model_dir)
        saver =  tf.train.import_meta_graph(model_checkpoint + '.meta')

        op_dict['train'] = graph.get_collection('train')[0]
        op_dict['input'] = graph.get_collection('input')[0]
        op_dict['output'] = graph.get_collection('output')[0]
        op_dict['global_step'] = graph.get_collection('global_step')[0]
        op_dict['fit'] =  graph.get_collection('fit') + graph.get_collection(tf.GraphKeys.UPDATE_OPS)
        op_dict['evaluate'] = graph.get_collection('evaluate')[0]
        op_dict['logits'] = graph.get_collection('logits')[0]
        op_dict['saver'] = saver

    return graph, op_dict, model_checkpoint
