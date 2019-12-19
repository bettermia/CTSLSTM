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
import pandas as pd
import json

hps = json.load(open('./hparam_files/HyperParameters.json', 'r'))

def prior_init(shape, dtype=None):
        priorMatrix = pd.read_csv(hps['priorMatrix_path']).values[:,1:].astype(np.float32)
        priorMatrix = tf.convert_to_tensor(priorMatrix)
        return priorMatrix

class ConvRNN2D(RNN):
    def __init__(self, cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if unroll:
            raise TypeError('Unrolling isn\'t possible with '
                            'convolutional RNNs.')
        if isinstance(cell, (list, tuple)):
            # The StackedConvRNN2DCells isn't implemented yet.
            raise TypeError('It is not possible at the moment to'
                            'stack convolutional cells.')
        super(ConvRNN2D, self).__init__(cell,
                                        return_sequences,
                                        return_state,
                                        go_backwards,
                                        stateful,
                                        unroll,
                                        **kwargs)
        self.input_spec = [InputSpec(ndim=5)]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        cell = self.cell

        output_shape = input_shape[:2] + (2, input_shape[3], cell.units)
        return output_shape

    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants:]
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if self.cell.data_format == 'channels_first':
                ch_dim = 1
            elif self.cell.data_format == 'channels_last':
                ch_dim = 3
            if not [spec.shape[ch_dim] for spec in self.state_spec] == state_size:
                raise ValueError(
                    'An initial_state was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'However `cell.state_size` is '
                    '{}'.format([spec.shape for spec in self.state_spec], self.cell.state_size))
        else:
            if self.cell.data_format == 'channels_first':
                self.state_spec = [InputSpec(shape=(None, dim, None, None))
                                   for dim in state_size]
            elif self.cell.data_format == 'channels_last':
                self.state_spec = [InputSpec(shape=(None, None, None, dim))
                                   for dim in state_size]
        if self.stateful:
            return None
        self.built = True

    def get_initial_state(self, inputs):
        # (samples, timesteps, channels, seriesNum, dim)
        initial_statetemp = K.zeros_like(inputs)
        # (samples, channels, seriesNum, dim)
        initial_statetemp = K.sum(initial_statetemp, axis=1)
        initial_state = initial_statetemp
        if inputs.shape[2]==1:
            for i in range(self.units-1):
                initial_state = tf.concat([initial_statetemp,initial_state],3)
            initial_state= tf.concat([initial_state,initial_state],1)
        else:
            initial_statetemp = K.sum(initial_statetemp, axis=3)
            initial_statetemp = K.expand_dims(initial_statetemp)
            initial_state = initial_statetemp
            for i in range(self.units-1):
                initial_state = tf.concat([initial_statetemp,initial_state],3)
        # print('initial_state')
        # print(initial_state.shape)



        # Fix for Theano because it needs
        # K.int_shape to work in call() with initial_state.
        # keras_shape = list(K.int_shape(inputs))
        # keras_shape.pop(1)
        # if K.image_data_format() == 'channels_first':
        #     indices = 2, 3
        # else:
        #     indices = 1, 2
        # for i, j in enumerate(indices):
        #     keras_shape[j] = conv_utils.conv_output_length(
        #         keras_shape[j],
        #         shape[i],
        #         padding=self.cell.padding,
        #         stride=self.cell.strides[i],
        #         dilation=self.cell.dilation_rate[i])
        # initial_state._keras_shape = keras_shape

        if hasattr(self.cell.state_size, '__len__'):
            return [initial_state for _ in self.cell.state_size]
        else:
            return [initial_state]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = self._standardize_args(
            inputs, initial_state, constants)

        if initial_state is None and constants is None:
            return super(ConvRNN2D, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = []
            for state in initial_state:
                try:
                    shape = K.int_shape(state)
                # Fix for Theano
                except TypeError:
                    shape = tuple(None for _ in range(K.ndim(state)))
                self.state_spec.append(InputSpec(shape=shape))

            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != K.is_keras_tensor(additional_inputs[0]):
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors')

        if K.is_keras_tensor(additional_inputs[0]):
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(ConvRNN2D, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output

        else:
            return super(ConvRNN2D, self).__call__(inputs, **kwargs)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        timesteps = K.int_shape(inputs)[1]

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            # print('output')
            # print(output.shape)
            return output

        # helper function
        def get_tuple_shape(nb_channels):
            result = list(state_shape)
            if self.cell.data_format == 'channels_first':
                result[1] = nb_channels
            elif self.cell.data_format == 'channels_last':
                result[3] = nb_channels
            else:
                raise KeyError
            return tuple(result)

        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, '__len__'):
                self.states = [K.zeros(get_tuple_shape(dim))
                               for dim in self.cell.state_size]
            else:
                self.states = [K.zeros(get_tuple_shape(self.cell.state_size))]
        elif states is None:
            if hasattr(self.cell.state_size, '__len__'):
                for state, dim in zip(self.states, self.cell.state_size):
                    K.set_value(state, np.zeros(get_tuple_shape(dim)))
            else:
                K.set_value(self.states[0],
                            np.zeros(get_tuple_shape(self.cell.state_size)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                                         'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != get_tuple_shape(dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str(get_tuple_shape(dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO: consider batch calls to `set_value`.
                K.set_value(state, value)

class ConvLSTM2DCell(Layer):
    def __init__(self, units,
                 #kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_first',
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(ConvLSTM2DCell, self).__init__(**kwargs)
        self.units = units
        #self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        seriesNum = input_shape[2]
        dataDim = input_shape[3]
        # self.kernel_initTwoParts = self.add_weight(shape=(dataDim, self.units * 2),
        #                               name='kernel_initTwoParts',
        #                               initializer=self.kernel_initializer,
        #                               regularizer=self.kernel_regularizer,
        #                               constraint=self.kernel_constraint)
        self.S_kernel = self.add_weight(shape=(seriesNum,seriesNum),name='S_kernel',
                                        initializer=prior_init,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.kernel = self.add_weight(shape=(dataDim, self.units*8),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 8),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:   
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units*2,), *args, **kwargs), 
                        initializers.Ones()((self.units*2,), *args, **kwargs), 
                        self.bias_initializer((self.units * 4,), *args, **kwargs), 
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 8,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.inner_kernel_i = self.kernel[:, :self.units]
        self.inner_kernel_f = self.kernel[:, self.units: self.units * 2]
        self.inner_kernel_o = self.kernel[:, self.units * 2:self.units * 3]
        self.inner_kernel_c = self.kernel[:, self.units * 3:self.units * 4]

        self.inter_kernel_i = self.kernel[:, self.units * 4:self.units * 5]
        self.inter_kernel_f = self.kernel[:, self.units*5: self.units * 6]
        self.inter_kernel_o = self.kernel[:, self.units * 6:self.units * 7]
        self.inter_kernel_c = self.kernel[:, self.units * 7:self.units * 8]    

        self.inner_recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.inner_recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.inner_recurrent_kernel_o = self.recurrent_kernel[:, self.units * 2:self.units * 3]
        self.inner_recurrent_kernel_c = self.recurrent_kernel[:, self.units * 3:self.units * 4]

        self.inter_recurrent_kernel_i = self.recurrent_kernel[:, self.units * 4:self.units * 5]
        self.inter_recurrent_kernel_f = self.recurrent_kernel[:, self.units*5: self.units * 6]
        self.inter_recurrent_kernel_o = self.recurrent_kernel[:, self.units * 6:self.units * 7]
        self.inter_recurrent_kernel_c = self.recurrent_kernel[:, self.units * 7:self.units * 8]
        if self.use_bias:
            self.inner_bias_i = self.bias[:self.units]            
            self.inner_bias_f = self.bias[self.units * 2:self.units * 3]           
            self.inner_bias_o = self.bias[self.units * 4:self.units * 5]            
            self.inner_bias_c = self.bias[self.units*6: self.units * 7]            
            self.inter_bias_i = self.bias[self.units: self.units * 2]           
            self.inter_bias_f = self.bias[self.units * 3:self.units * 4]            
            self.inter_bias_o = self.bias[self.units * 5:self.units * 6]           
            self.inter_bias_c = self.bias[self.units * 7:self.units * 8]            
            self.inner_bias_i = tf.reshape(self.inner_bias_i,(1,1,self.units))
            self.inner_bias_f = tf.reshape(self.inner_bias_f,(1,1,self.units))
            self.inner_bias_o = tf.reshape(self.inner_bias_o,(1,1,self.units))
            self.inner_bias_c = tf.reshape(self.inner_bias_c,(1,1,self.units))
            self.inter_bias_i = tf.reshape(self.inter_bias_i,(1,1,self.units))
            self.inter_bias_f = tf.reshape(self.inter_bias_f,(1,1,self.units))
            self.inter_bias_o = tf.reshape(self.inter_bias_o,(1,1,self.units))
            self.inter_bias_c = tf.reshape(self.inter_bias_c,(1,1,self.units))

        else:
            self.inner_bias_i = None
            self.inner_bias_f = None
            self.inner_bias_o = None
            self.inner_bias_c = None
            self.inter_bias_i = None
            self.inter_bias_f = None
            self.inter_bias_o = None
            self.inter_bias_c = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        seriesNum = inputs.shape[2]
        dataDim = inputs.shape[3]
        channels = inputs.shape[1]
        # print('=========')
        # print(h_tm1.shape)
        # print(c_tm1.shape)
        if channels==1:
            inputs_inner = tf.reshape(inputs,(-1,seriesNum,dataDim))
            inputs_inter = tf.reshape(inputs,(-1,seriesNum,dataDim))
            # h_tm1_inner = h_tm1
            # h_tm1_inter = h_tm1
            # c_tm1_inner = c_tm1
            # c_tm1_inter = c_tm1
            # h_tm1_inner = tf.reshape(h_tm1_inner,(-1,seriesNum,dataDim))
            # h_tm1_inter = tf.reshape(h_tm1_inter,(-1,seriesNum,dataDim))
            # c_tm1_inner = tf.reshape(c_tm1_inner,(-1,seriesNum,dataDim))
            # c_tm1_inter = tf.reshape(c_tm1_inter,(-1,seriesNum,dataDim))
        else:
            inputs_inner = inputs[:,0:1,:,:]
            inputs_inner = tf.reshape(inputs_inner,(-1,seriesNum,dataDim))
            inputs_inter = inputs[:,1:2,:,:]
            inputs_inter = tf.reshape(inputs_inter,(-1,seriesNum,dataDim))
            # h_tm1_inner = h_tm1[:,0:1,:,:]
            # h_tm1_inter = h_tm1[:,1:2,:,:]
            # c_tm1_inner = c_tm1[:,0:1,:,:]
            # c_tm1_inter = c_tm1[:,1:2,:,:]
            # h_tm1_inner = tf.reshape(h_tm1_inner,(-1,seriesNum,dataDim))
            # h_tm1_inter = tf.reshape(h_tm1_inter,(-1,seriesNum,dataDim))
            # c_tm1_inner = tf.reshape(c_tm1_inner,(-1,seriesNum,dataDim))
            # c_tm1_inter = tf.reshape(c_tm1_inter,(-1,seriesNum,dataDim))
        h_tm1_inner = h_tm1[:,0:1,:,:]
        h_tm1_inter = h_tm1[:,1:2,:,:]
        c_tm1_inner = c_tm1[:,0:1,:,:]
        c_tm1_inter = c_tm1[:,1:2,:,:]
        h_tm1_inner = tf.reshape(h_tm1_inner,(-1,seriesNum,self.units))
        h_tm1_inter = tf.reshape(h_tm1_inter,(-1,seriesNum,self.units))
        c_tm1_inner = tf.reshape(c_tm1_inner,(-1,seriesNum,self.units))
        c_tm1_inter = tf.reshape(c_tm1_inter,(-1,seriesNum,self.units))
        # print('dot shape')
        # print(inputs_inter.shape)
        # print(self.S_kernel.shape)
        inputs_inter = K.dotSelf(self.S_kernel,inputs_inter)
        # print(inputs_inter.shape)
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            #print('start drop')
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs_inner),
                self.dropout,
                training=training,
                count=8)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            #print('start recurrent_dropout')
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(c_tm1_inter),
                self.recurrent_dropout,
                training=training,
                count=8)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if 0 < self.dropout < 1.:
            inner_inputs_i = inputs_inner * dp_mask[0]
            inner_inputs_f = inputs_inner * dp_mask[1]
            inner_inputs_c = inputs_inner * dp_mask[2]
            inner_inputs_o = inputs_inner * dp_mask[3]

            inter_inputs_i = inputs_inter* dp_mask[4]
            inter_inputs_f = inputs_inter* dp_mask[5]
            inter_inputs_c = inputs_inter* dp_mask[6]
            inter_inputs_o = inputs_inter* dp_mask[7]
        else:
            inner_inputs_i = inputs_inner
            inner_inputs_f = inputs_inner
            inner_inputs_c = inputs_inner
            inner_inputs_o = inputs_inner

            inter_inputs_i = inputs_inter
            inter_inputs_f = inputs_inter
            inter_inputs_c = inputs_inter
            inter_inputs_o = inputs_inter
        if 0 < self.recurrent_dropout < 1.:
            inner_h_tm1_i = h_tm1_inner * rec_dp_mask[0]
            inner_h_tm1_f = h_tm1_inner * rec_dp_mask[1]
            inner_h_tm1_c = h_tm1_inner * rec_dp_mask[2]
            inner_h_tm1_o = h_tm1_inner * rec_dp_mask[3]

            inter_h_tm1_i = h_tm1_inter * rec_dp_mask[4]
            inter_h_tm1_f = h_tm1_inter * rec_dp_mask[5]
            inter_h_tm1_c = h_tm1_inter * rec_dp_mask[6]
            inter_h_tm1_o = h_tm1_inter * rec_dp_mask[7]
        else:
            inner_h_tm1_i = h_tm1_inner
            inner_h_tm1_f = h_tm1_inner
            inner_h_tm1_c = h_tm1_inner
            inner_h_tm1_o = h_tm1_inner

            inter_h_tm1_i = h_tm1_inter 
            inter_h_tm1_f = h_tm1_inter 
            inter_h_tm1_c = h_tm1_inter 
            inter_h_tm1_o = h_tm1_inter

        x_i_inner = K.dot(inner_inputs_i,self.inner_kernel_i)
        x_f_inner = K.dot(inner_inputs_f,self.inner_kernel_f)
        x_o_inner = K.dot(inner_inputs_o,self.inner_kernel_o)
        x_c_inner = K.dot(inner_inputs_c,self.inner_kernel_c)

        x_i_inter = K.dot(inter_inputs_i,self.inter_kernel_i)
        x_f_inter = K.dot(inter_inputs_f,self.inter_kernel_f)
        x_o_inter = K.dot(inter_inputs_o,self.inter_kernel_o)
        x_c_inter = K.dot(inter_inputs_c,self.inter_kernel_c)

        h_i_inner = K.dot(inner_h_tm1_i,self.inner_recurrent_kernel_i)
        h_f_inner = K.dot(inner_h_tm1_f,self.inner_recurrent_kernel_f)
        h_o_inner = K.dot(inner_h_tm1_o,self.inner_recurrent_kernel_o)
        h_c_inner = K.dot(inner_h_tm1_c,self.inner_recurrent_kernel_c)

        h_i_inter = K.dot(inter_h_tm1_i,self.inter_recurrent_kernel_i)
        h_f_inter = K.dot(inter_h_tm1_f,self.inter_recurrent_kernel_f)
        h_o_inter = K.dot(inter_h_tm1_o,self.inter_recurrent_kernel_o)
        h_c_inter = K.dot(inter_h_tm1_c,self.inter_recurrent_kernel_c)
        if self.use_bias:
            # x_i_inner = K.bias_add(x_i_inner, self.inner_bias_i)
            # x_f_inner = K.bias_add(x_f_inner, self.inner_bias_f)
            # x_o_inner = K.bias_add(x_o_inner, self.inner_bias_o)
            # x_c_inner = K.bias_add(x_c_inner, self.inner_bias_c)

            # x_i_inter = K.bias_add(x_i_inter, self.inter_bias_i)
            # x_f_inter = K.bias_add(x_f_inter, self.inter_bias_f)
            # x_o_inter = K.bias_add(x_o_inter, self.inter_bias_o)
            # x_c_inter = K.bias_add(x_c_inter, self.inter_bias_c)

            x_i_inner = x_i_inner+self.inner_bias_i
            x_f_inner = x_f_inner+self.inner_bias_f
            x_o_inner = x_o_inner+self.inner_bias_o
            x_c_inner = x_c_inner+self.inner_bias_c

            x_i_inter = x_i_inter+self.inter_bias_i
            x_f_inter = x_f_inter+self.inter_bias_f
            x_o_inter = x_o_inter+self.inter_bias_o
            x_c_inter = x_c_inter+self.inter_bias_c
        inner_i = self.recurrent_activation(x_i_inner + h_i_inner)
        inner_f = self.recurrent_activation(x_f_inner + h_f_inner)
        inner_o = self.recurrent_activation(x_o_inner + h_o_inner)
        inner_c = inner_f * c_tm1_inner + inner_i * self.activation(x_c_inner + h_c_inner)
        inner_h = inner_o * self.activation(inner_c)

        # temp = inner_o + h_o_inter
        # temp2=  inner_o + x_o_inter
        inter_i = self.recurrent_activation(x_i_inter + h_i_inter)
        inter_f = self.recurrent_activation(x_f_inter + h_f_inter)
        inter_o = self.recurrent_activation(x_o_inter + h_o_inter)
        inter_c = inter_f * c_tm1_inter + inter_i * self.activation(x_c_inter + h_c_inter)
        inter_h = inter_o * self.activation(inter_c)
        inner_h = tf.reshape(inner_h,(-1,1,inner_h.shape[1],inner_h.shape[2]))
        inter_h = tf.reshape(inter_h,(-1,1,inter_h.shape[1],inter_h.shape[2]))
        inner_c = tf.reshape(inner_c,(-1,1,inner_c.shape[1],inner_c.shape[2]))
        inter_c = tf.reshape(inter_c,(-1,1,inter_c.shape[1],inter_c.shape[2]))

        # print('concat')
        # print(inputs.shape)
        # print(inter_i.shape)
        # print(x_c_inter.shape)
        # print(h_c_inter.shape)
        # print(inter_o.shape)
        # print(inner_h.shape)
        # print(inter_h.shape)
        # print(inner_c.shape)
        # print(inter_c.shape)
        h = tf.concat([inner_h,inter_h],1)
        c = tf.concat([inner_c,inter_c],1)
        # print('hshape')
        # print(h.shape)
        # print(c.shape)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  # 'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvLSTM2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MultiChannelLSTM(ConvRNN2D):
    @interfaces.legacy_convlstm2d_support
    def __init__(self, units,
                 # kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = ConvLSTM2DCell(units=units,
                              # kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              data_format=data_format,
                              dilation_rate=dilation_rate,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              unit_forget_bias=unit_forget_bias,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout)
        super(MultiChannelLSTM, self).__init__(cell,
                                         return_sequences=return_sequences,
                                         go_backwards=go_backwards,
                                         stateful=stateful,
                                         **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(MultiChannelLSTM, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    # @property
    # def kernel_size(self):
    #     return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'units': self.units,
                  # 'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(MultiChannelLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
