#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 19:35:13 2018

@author: lukas
"""



import numpy
import numpy as np
import warnings
from keras.layers import Convolution3D, Input, merge, RepeatVector, Activation
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras import activations, initializers, regularizers
from keras.engine import Layer, InputSpec
#from keras.utils.np_utils import conv_output_length
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.engine.topology import Layer
import functools
import tensorflow as tf
import pickle
import time
from keras.layers.merge import concatenate, add
from keras.activations import softmax, relu
from keras.layers import Activation
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras.initializers import Orthogonal



class Deconvolution3D(Layer):
    
    def __init__(self, nb_filter, kernel_dims, output_shape, subsample, **kwargs):
        self.nb_filter = nb_filter
        self.kernel_dims = kernel_dims
        self.strides = (1, ) + subsample + (1, )
        self.output_shape_ = output_shape
        assert K.backend() == 'tensorflow'
        super(Deconvolution3D, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) == 5
        self.W = self.add_weight(shape=self.kernel_dims + (self.nb_filter, input_shape[4], ),
                                 initializer='glorot_uniform',
                                 name='{}_W'.format(self.name),
                                 trainable=True)
        self.b = self.add_weight(shape=(1, 1, 1, self.nb_filter,), 
                                 initializer='zero', 
                                 name='{}_b'.format(self.name),
                                 trainable=True)
        super(Deconvolution3D, self).build(input_shape) 

    def call(self, x, mask=None):
        return tf.nn.conv3d_transpose(x, self.W, output_shape=self.output_shape_,
                                      strides=self.strides, padding='SAME', name=self.name) + self.b

    def compute_output_shape(self, input_shape):
        return (input_shape[0], ) + self.output_shape_[1:]


from keras import backend as K
from keras.engine import Layer




def downward_layer(input_layer, n_convolutions, n_output_channels):
    inl = input_layer
    for _ in range(n_convolutions-1):
        inl = PReLU()(Convolution3D(n_output_channels // 2, (5, 5, 5), padding='same', data_format="channels_last",kernel_initializer=Orthogonal())(inl))
    conv = Convolution3D(n_output_channels // 2, (5, 5, 5), padding='same', data_format="channels_last")(inl)
    add_node = add([conv, input_layer])
    downsample = Convolution3D(n_output_channels, (2,2,2), strides=(2,2,2),kernel_initializer=Orthogonal())(add_node)
    prelu = PReLU()(downsample)
    return prelu, add_node

def upward_layer(input0 ,input1, n_convolutions, n_output_channels):
    merged = concatenate([input0, input1], axis=4)
    inl = merged
    for _ in range(n_convolutions-1):
        inl = PReLU()(Convolution3D(n_output_channels * 4, (5, 5, 5), padding='same', data_format="channels_last",kernel_initializer=Orthogonal())(inl))
    conv = Convolution3D(n_output_channels * 4, (5, 5, 5), padding='same', data_format="channels_last")(inl)
    add_node = add([conv, merged])
    shape = add_node.get_shape().as_list()
    new_shape = (1, shape[1] * 2, shape[2] * 2, shape[3] * 2, n_output_channels)
    upsample = Deconvolution3D(n_output_channels, (4,4,4), new_shape, subsample=(2,2,2))(add_node)
    return PReLU()(upsample)

# CHANGE IMAGE DIMENSIONS, INPUT AND OUTPUT. CHANGING DIMENSIONS WILL REQUIRE TO ADAPT ALL DIMENSION REDUCTIONS AND UPSCALINGS

# HERE THEY GO FROM 128,128,64, DOWN TO 8,8,4 , UP AGAIN TO 128,128,64
# I WANT TO GO FROM 181,217,298 DOWN TO WHATEVER, UP AGAIN TO 181,217,298  (THESE ARE THE DIMENSIONS OF THE resampled TPM to 1mm isotropic voxel size)

'''
output_classes = 2

a = 2 # these channels are connected, must change accordingly. These are the input channels that get added to the first conv layer.

# Layer 1
input_layer = Input(shape=(176, 208, 288, 1), name='data')
conv_1 = Convolution3D(a, (5, 5, 5), padding='same', data_format="channels_last")(input_layer)
repeat_1 = concatenate([input_layer] * a)
add_1 = add([conv_1, repeat_1])
prelu_1_1 = PReLU()(add_1)
downsample_1 = Convolution3D(2, (2,2,2), strides=(2,2,2))(prelu_1_1)
prelu_1_2 = PReLU()(downsample_1)

# Layer 2,3,4
out2, left2 = downward_layer(prelu_1_2, 2, 4)
out3, left3 = downward_layer(out2, 2,2)
out4, left4 = downward_layer(out3, 2,2)

# Layer 5
conv_5_1 = Convolution3D(2, (5, 5, 4), padding='same', data_format="channels_last")(out4)
prelu_5_1 = PReLU()(conv_5_1)
conv_5_2 = Convolution3D(2, (5, 5, 4), padding='same', data_format="channels_last")(prelu_5_1)
prelu_5_2 = PReLU()(conv_5_2)
conv_5_3 = Convolution3D(2, (5, 5, 4), padding='same', data_format="channels_last")(prelu_5_2)
add_5 = add([conv_5_3, out4])
prelu_5_1 = PReLU()(add_5)


downsample_5 = Deconvolution3D(14, (2,2,2), (1, 22, 26, 36, 14), subsample=(2,2,2))(prelu_5_1)
prelu_5_2 = PReLU()(downsample_5)

#Layer 6,7,8
out6 = upward_layer(prelu_5_2, left4, 2,4)
out7 = upward_layer(out6, left3, 2, 2)
out8 = upward_layer(out7, left2, 2, 1)

#Layer 9
merged_9 = concatenate([out8, add_1], axis=4)
conv_9_1 = Convolution3D(1, (5, 5, 5), padding='same', data_format="channels_last")(merged_9)
add_9 = add([conv_9_1, merged_9])
conv_9_2 = Convolution3D(output_classes, (1, 1, 1), padding='same', data_format="channels_last")(add_9)

softmax_layer =  Activation(softmax)(conv_9_2)

#softmax = Softmax()(conv_9_2)

model = Model(input_layer, softmax_layer)

model.summary(line_length=113)

from keras.utils import plot_model
plot_model(model, 'V-Net_tiny2.png', show_shapes=True)
'''

#%%
output_classes = 2
a = 16  # goes from a down to a/16. Minimum value for a = 32 (concatenate at the beginning needs at least 2 input images to concatenate...) Atlthough can change that inde[endently]
b = 4  # this establishes how many input images are concatenated at the start. Must be synchronized with the last layer too.
# Layer 1



input_layer = Input(shape=(176, 208, 288, 1), name='data')
conv_1 = Convolution3D( (a/16)+4 , (5, 5, 5), padding='same', data_format="channels_last",kernel_initializer=Orthogonal())(input_layer)  
repeat_1 = concatenate([input_layer] * ((a/16) +b) )
add_1 = add([conv_1, repeat_1])
prelu_1_1 = PReLU()(add_1)
downsample_1 = Convolution3D(a/8, (2,2,2), strides=(2,2,2))(prelu_1_1)
prelu_1_2 = PReLU()(downsample_1)

# Layer 2,3,4
out2, left2 = downward_layer(prelu_1_2, 2, a/4)
out3, left3 = downward_layer(out2, 2, a/2)
out4, left4 = downward_layer(out3, 2, a)

# Layer 5
conv_5_1 = Convolution3D(a/2, (5, 5, 4), padding='same', data_format="channels_last",kernel_initializer=Orthogonal())(out4) # originally set to a
prelu_5_1 = PReLU()(conv_5_1)
conv_5_2 = Convolution3D(a/2, (5, 5, 4), padding='same', data_format="channels_last",kernel_initializer=Orthogonal())(prelu_5_1) # originally set to a
prelu_5_2 = PReLU()(conv_5_2)
conv_5_3 = Convolution3D(a, (5, 5, 4), padding='same', data_format="channels_last",kernel_initializer=Orthogonal())(prelu_5_2)   # needs to match output of out4
add_5 = add([conv_5_3, out4])
prelu_5_1 = PReLU()(add_5)
downsample_5 = Deconvolution3D(a/2, (2,2,2), (1, 22, 26, 36, a/2), subsample=(2,2,2))(prelu_5_1)
prelu_5_2 = PReLU()(downsample_5)

#Layer 6,7,8
out6 = upward_layer(prelu_5_2, left4, 3, a/4)
out7 = upward_layer(out6, left3, 3, a/8)
out8 = upward_layer(out7, left2, 2, a/16)

#Layer 9
merged_9 = concatenate([out8, add_1], axis=4)
conv_9_1 = Convolution3D(a/8 + b, (5, 5, 5), padding='same', data_format="channels_last")(merged_9)
add_9 = add([conv_9_1, merged_9])
conv_9_2 = Convolution3D(output_classes, (1, 1, 1), padding='same', data_format="channels_last")(add_9)

softmax_layer =  Activation(softmax)(conv_9_2)

model = Model(input_layer, softmax_layer)

model.summary(line_length=113)

from keras.utils import plot_model
plot_model(model, 'V-Net_tiny2.png', show_shapes=True)


#%%



def downward_layer_relu(input_layer, n_convolutions, n_output_channels):
    inl = input_layer
    for _ in range(n_convolutions-1):
        inl = Activation('relu')(Convolution3D(n_output_channels // 2, (5, 5, 5), padding='same', data_format="channels_last")(inl))
    conv = Convolution3D(n_output_channels // 2, (5, 5, 5), padding='same', data_format="channels_last")(inl)
    add_node = add([conv, input_layer])
    downsample = Convolution3D(n_output_channels, (2,2,2), strides=(2,2,2))(add_node)
    prelu = Activation('relu')(downsample)
    return prelu, add_node

def upward_layer_relu(input0 ,input1, n_convolutions, n_output_channels):
    merged = concatenate([input0, input1], axis=4)
    inl = merged
    for _ in range(n_convolutions-1):
        inl = Activation('relu')(Convolution3D(n_output_channels * 4, (5, 5, 5), padding='same', data_format="channels_last")(inl))
    conv = Convolution3D(n_output_channels * 4, (5, 5, 5), padding='same', data_format="channels_last")(inl)
    add_node = add([conv, merged])
    shape = add_node.get_shape().as_list()
    new_shape = (1, shape[1] * 2, shape[2] * 2, shape[3] * 2, n_output_channels)
    upsample = Deconvolution3D(n_output_channels, (4,4,4), new_shape, subsample=(2,2,2))(add_node)
    return Activation('relu')(upsample)

output_classes = 2
a = 64  # goes from a down to a/16. Minimum value for a = 32 (concatenate at the beginning needs at least 2 input images to concatenate...) Atlthough can change that inde[endently]
b = 4  # this establishes how many input images are concatenated at the start. Must be synchronized with the last layer too.
# Layer 1
input_layer = Input(shape=(176, 208, 288, 1), name='data')
conv_1 = Convolution3D( (a/16)+4 , (5, 5, 5), padding='same', data_format="channels_last")(input_layer)  
repeat_1 = concatenate([input_layer] * ((a/16) +b) )
add_1 = add([conv_1, repeat_1])
prelu_1_1 = Activation('relu')(add_1)
downsample_1 = Convolution3D(a/8, (2,2,2), strides=(2,2,2))(prelu_1_1)
prelu_1_2 = Activation('relu')(downsample_1)

# Layer 2,3,4
out2, left2 = downward_layer_relu(prelu_1_2, 2, a/4)
out3, left3 = downward_layer_relu(out2, 2, a/2)
out4, left4 = downward_layer_relu(out3, 2, a)

# Layer 5
conv_5_1 = Convolution3D(a/2, (5, 5, 4), padding='same', data_format="channels_last")(out4) # originally set to a
prelu_5_1 = Activation('relu')(conv_5_1)
conv_5_2 = Convolution3D(a/2, (5, 5, 4), padding='same', data_format="channels_last")(prelu_5_1) # originally set to a
prelu_5_2 = Activation('relu')(conv_5_2)
conv_5_3 = Convolution3D(a, (5, 5, 4), padding='same', data_format="channels_last")(prelu_5_2)   # needs to match output of out4
add_5 = add([conv_5_3, out4])
prelu_5_1 = Activation('relu')(add_5)
downsample_5 = Deconvolution3D(a/2, (2,2,2), (1, 22, 26, 36, a/2), subsample=(2,2,2))(prelu_5_1)
prelu_5_2 = Activation('relu')(downsample_5)

#Layer 6,7,8
out6 = upward_layer_relu(prelu_5_2, left4, 3, a/4)
out7 = upward_layer_relu(out6, left3, 3, a/8)
out8 = upward_layer_relu(out7, left2, 2, a/16)

#Layer 9
merged_9 = concatenate([out8, add_1], axis=4)
conv_9_1 = Convolution3D(a/8 + b, (5, 5, 5), padding='same', data_format="channels_last")(merged_9)
add_9 = add([conv_9_1, merged_9])
conv_9_2 = Convolution3D(output_classes, (1, 1, 1), padding='same', data_format="channels_last")(add_9)

softmax_layer =  Activation(softmax)(conv_9_2)

model = Model(input_layer, softmax_layer)

model.summary(line_length=113)

from keras.utils import plot_model
plot_model(model, 'V-Net_tiny2.png', show_shapes=True)


#%%##


#from keras.optimizers import RMSprop
#rmsprop = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-8)


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=2e-5), metrics=['accuracy'])

smooth = 1e-5

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=2):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice

model.compile(loss=dice_coef_multilabel, optimizer=Adam(lr=1e-8), metrics=['accuracy'])

'''
def soft_dice_numpy(y_pred, y_true, eps=1e-7):

    axes = tuple(range(2, len(y_pred.shape)))
    intersect = K.sum(y_pred * y_true, axes)
    denom = K.sum(y_pred + y_true, axes)
    return   K.mean((2. *intersect / (denom + eps)))


model.compile(optimizer=Adam(lr=1e-2), loss=soft_dice_numpy, metrics=['accuracy'])


smooth = 1e-05

def dice_coef(y_true, y_pred):
    
    y_true_f = y_true[:,:,:,:,1]
    y_pred_f = y_pred[:,:,:,:,1]
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


#model_dice = dice_loss(smooth=1e-5, thresh=0.5)

model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=['accuracy'])

'''
######################### DATA  ####################################
 
img = ['/home/lukas/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/src0011s0006t01 (copy).nii']
#label_address = '/home/lukas/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/srsegmentsMixtureBinary.nii'

label_address = '/home/lukas/Documents/projects/TPM/srTPM_GM.nii'

import nibabel as nib

data = nib.load(img[0]).get_data()
aff = nib.load(img[0]).affine

label = nib.load(label_address).get_data()

x = data.reshape((1,176,208,288,1))

from keras.utils import to_categorical


thr = 0.1
label[label > thr] = 1
label[label <= thr] = 0
pred = nib.Nifti1Image(label, aff)
print 'saving prediction'    
nib.save(pred, 'target_GM2.nii')

y = to_categorical(label, output_classes)
y = np.array([y])
y.shape

'''
sample_weight = np.zeros((1,176,208,288,2))
sample_weight[:,:,:,:, 0] += 1
sample_weight[:,:,:,:, 1] += 1
sample_weight.shape
sample_weight[:,:,:,:,1]
'''

'''
x_null = np.zeros((y.shape))
x_true = label.reshape((y.shape))
y_t = to_categorical(x_true, 2)

x_null_tf = tf.convert_to_tensor(x_null, np.float32)
x_true_tf = tf.convert_to_tensor(x_true, np.float32)
y_true_tf = tf.convert_to_tensor(y, np.float32)

loss_dice = dice_coef_multilabel(x_null_tf, y_true_tf)

sess = tf.InteractiveSession()
print(loss_dice.eval())
sess.close()
#model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
'''

#%%###################### TRAINING ###########################################

epoch = 20

model.evaluate(x,y)
for i in range(14, 14+epoch):
    print i
    y_pred = model.predict(x)
    class_pred = np.array(np.argmax(y_pred, axis=4), 'float32')
    pred = nib.Nifti1Image(class_pred[0]*255, aff)
    print 'saving prediction'    
    nib.save(pred, 'tiny2_DICE_GM_pred_epoch{}.nii'.format(i))
    model.fit(x, y, epochs=25, verbose=1)#, class_weight = sample_weight)
    #model.train_on_batch(x,y)



def dice_completeImages(y_true, y_pred):
    return(2*np.sum(np.multiply(y_pred>0,y_true>0))/float(np.sum(y_pred>0)+np.sum(y_true>0)))
    
    
class_true = np.array(np.argmax(y, axis=4), 'float32')
dice_completeImages(class_true, class_pred)

