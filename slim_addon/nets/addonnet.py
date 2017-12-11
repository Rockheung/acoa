# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_a.default_image_size = 224

def addon_net(input_, num_of_filer, lv1_num_classes, lv1_classname, 
                is_training=True, 
                dropout_keep_prob=0.5, 
                spatial_squeeze = True, 
                fc_conv_padding = 'VALID'):
  net = slim.repeat(input_, 3, slim.conv2d, num_of_filer[0], [3, 3], scope='conv1')
  net = slim.max_pool2d(net, [2, 2], scope='pool1')
  net = slim.repeat(net, 3, slim.conv2d, num_of_filer[1], [3, 3], scope='conv2')
  net = slim.max_pool2d(net, [2, 2], scope='pool2')
  # Use conv2d instead of fully_connected layers.
  net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc1')
  net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                              scope='dropout1')
  net = slim.conv2d(net, 4096, [1, 1], scope='fc2')
  # if global_pool:
  #   net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='addon/Bottom/global_pool')
  #   end_points['global_pool'] = net
  if lv1_num_classes[lv1_classname]:
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                scope='dropout2')
    net = slim.conv2d(net, lv1_num_classes[lv1_classname], [1, 1],
                              activation_fn=None,
                              normalizer_fn=None,
                              scope='fc3')
    if spatial_squeeze and lv1_num_classes[lv1_classname] is not None:
      net = tf.squeeze(net, [1, 2], name='fc3/squeezed')
    return net


def vgg_16(inputs,	
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """

  # data =  {'Outer': ['cardigan', 'jacket', 'coat'], 'Bottom': ['mini-skirt', 'short_pants', 'skirt', 'long_pants'], 'Dress': ['short_dress', 'long_dress'], 'Top': ['polo_shirt', 'T-shirt,tank-top', 'long-sleeved,sweat_shirt', 'dress_shirt'], 'Shoe': ['slipper, sandal', 'gentleman shoe', 'snikers', 'ladys shoe', 'boot', 'baby shoe', 'running shoe'], 'Suit': ['suit'], 'Hat': ['cap', 'boater, Bucket', 'flat', 'beanie']}
  # for i in data:
  #     num_classes[i] = len(data[i])
  # print num_classes
  #    => {'Outer': 3, 'Bottom': 4, 'Hat': 4, 'Top': 4, 'Shoe': 7, 'Suit': 1, 'Dress': 2}
  lv1_num_classes = {'Outer': 3, 'Bottom': 4, 'Hat': 4, 'Top': 4, 'Shoe': 7, 'Suit': 1, 'Dress': 2}

  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      to_addon = net

      with tf.variable_scope('addon'):
        with tf.variable_scope('Dress'):
          a1 = addon_net(to_addon, [128, 64], lv1_num_classes, 'Dress')
        with tf.variable_scope('Hat'):
          a2 = addon_net(to_addon, [128, 64], lv1_num_classes, 'Hat')
        with tf.variable_scope('Outer'):
          a3 = addon_net(to_addon, [128, 64], lv1_num_classes, 'Outer')
        with tf.variable_scope('Top'):
          a4 = addon_net(to_addon, [128, 64], lv1_num_classes, 'Top')  
        with tf.variable_scope('Bottom'):
          a5 = addon_net(to_addon, [128, 64], lv1_num_classes, 'Bottom')
        with tf.variable_scope('Suit'):
          a6 = addon_net(to_addon, [128, 64], lv1_num_classes, 'Suit')  
        with tf.variable_scope('Shoe'):
          a7 = addon_net(to_addon, [128, 64], lv1_num_classes, 'Shoe')
        from_addon = tf.concat([a1, a2, a3, a4, a5, a6, a7], 1, name='concat')    


      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')

      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze and num_classes is not None:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, from_addon, end_points
vgg_16.default_image_size = 224


  # with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
  #   end_points_collection = sc.original_name_scope + '_end_points'
  #   # Collect outputs for conv2d, fully_connected and max_pool2d.
  #   with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
  #                       outputs_collections=end_points_collection):
  #     net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
  #     net = slim.max_pool2d(net, [2, 2], scope='pool1')
  #     net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
  #     net = slim.max_pool2d(net, [2, 2], scope='pool2')
  #     net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
  #     net = slim.max_pool2d(net, [2, 2], scope='pool3')


  #     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
  #     net = slim.max_pool2d(net, [2, 2], scope='pool4')

  #     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
  #     net = slim.max_pool2d(net, [2, 2], scope='pool5')

  #     # Use conv2d instead of fully_connected layers.
  #     net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
  #     net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
  #                        scope='dropout6')
  #     net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
  #     # Convert end_points_collection into a end_point dict.
  #     end_points = slim.utils.convert_collection_to_dict(end_points_collection)
  #     if global_pool:
  #       net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
  #       end_points['global_pool'] = net
  #     if num_classes:
  #       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
  #                          scope='dropout7')
  #       net = slim.conv2d(net, num_classes, [1, 1],
  #                         activation_fn=None,
  #                         normalizer_fn=None,
  #                         scope='fc8')
  #       if spatial_squeeze and num_classes is not None:
  #         net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
  #       end_points[sc.name + '/fc8'] = net
  #     return net, end_points
#vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19
