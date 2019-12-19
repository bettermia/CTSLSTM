"""Convolutional-recurrent layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import activations, initializers, regularizers,constraints
from keras.layers.recurrent import _generate_dropout_mask

import tensorflow as tf
import numpy as np
import warnings
from keras.engine import InputSpec, Layer
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.legacy.layers import Recurrent
from keras.layers.recurrent import RNN
from keras.utils.generic_utils import has_arg

class Fusion(Layer):
    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Fusion, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        # print('=======================')
        # print('build inputshape')
        # print(input_shape)
        # print('============================')
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        seriesNum = input_shape[2]
        self.Theta = self.add_weight(shape=(1, input_dim*2),
                                      initializer=self.kernel_initializer,
                                      name='Theta',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel = self.add_weight(shape=(seriesNum*input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.Theta1= self.Theta[:,:input_dim]
        self.Theta2 = self.Theta[:,input_dim:2*input_dim]
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        input_dim = inputs.shape[-1]
        seriesNum = inputs.shape[2]
        inner = inputs[:,0:1,:,:]
        inter = inputs[:,1:2,:,:]
        inner = K.sum(inner, axis=1)
        inter = K.sum(inter, axis=1)
        fusion = inner*self.Theta1+inter*self.Theta2
        fusion = tf.reshape(fusion,(-1,seriesNum*input_dim))
        output = K.dot(fusion, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        # print('DenseOutshape=====')
        # print(output.shape)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = (input_shape[0],self.units)
        # print('=====DenseOutshape')
        # print(output_shape)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Fusion, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
