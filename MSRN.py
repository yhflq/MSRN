import keras
from keras import regularizers
from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, MaxPooling2D,Reshape,DepthwiseConv2D, BatchNormalization,GlobalAveragePooling2D,Activation
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Input,Lambda,Concatenate,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from tensorflow.keras.callbacks import TensorBoard
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import random
from operator import truediv
from plotly.offline import init_notebook_mode

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import scipy.io as sio
import os
import spectral
import glob
from sklearn.preprocessing import MinMaxScaler


def squeeze_excitation_layer(input_layer,filter_sq,filter_ex):
    print(input_layer.shape)
    squeeze=GlobalAveragePooling2D()(input_layer)
    print(squeeze.shape)
    excitation=Reshape((1,1,filter_ex))(squeeze)
    y = tf.keras.layers.Conv1D(
filter_sq, 5, activation='relu',padding='SAME')(excitation)
    y=Dense(units=filter_ex,activation='sigmoid')(y)
    y=Activation('sigmoid')(y)
    scale=input_layer*y
    return scale
 #  ''' 


def SE_Conv_moule_1(input_layer):
    Conv_layer1=Conv2D(filters=192,kernel_size=(1,1),padding='same',activation='relu')(input_layer)
    Conv_layer1=BatchNormalization()(Conv_layer1)
    output_layer=squeeze_excitation_layer(Conv_layer1,192,192)
    return output_layer


def SE_Conv_moule_2(input_layer):
    Conv_layer1=Conv2D(filters=192,kernel_size=(3,3),padding='same',activation='relu')(input_layer)
    Conv_layer1=BatchNormalization()(Conv_layer1)
    output_layer=squeeze_excitation_layer(Conv_layer1,192,192)
    return output_layer


def SE_Conv_moule_3(input_layer,a):
    Conv_layer1=Conv2D(filters=a,kernel_size=(3,3),padding='same',activation='relu')(input_layer)
    Conv_layer1=BatchNormalization()(Conv_layer1)
    output_layer=squeeze_excitation_layer(Conv_layer1,a,a)
    return output_layer


def SE_Res_module(input_layer,a):
    layer1=BatchNormalization()(input_layer)
    layer2=Conv2D(filters=a,kernel_size=(3,3),padding='same',activation='relu')(layer1)
    layer3=BatchNormalization()(layer2)
    layer4=Conv2D(filters=a,kernel_size=(3,3),padding='same')(layer3)
    layer5=BatchNormalization()(layer4)
    layer6=squeeze_excitation_layer(layer5,a,a)
    layer7=tf.add(layer6,layer1)
    layer8=Activation('relu')(layer7)
    output_layer=BatchNormalization()(layer8)
    return output_layer



def attention_moudle(input_layer,filter,a,b): 
    layer1=Conv2D(filters=filter,kernel_size=(3,3),strides=1,padding='same')(input_layer)
    conv_11=DepthwiseConv2D(kernel_size=(1,1))(layer1)
    conv_12=DepthwiseConv2D(kernel_size=(1,1))(layer1)
    conv11_RES=Reshape((conv_11.shape[1]*conv_11.shape[2],conv_11.shape[3]))(conv_11)
    conv12_RES=Reshape((conv_12.shape[1]*conv_12.shape[2],conv_12.shape[3]))(conv_12)
    layer11=tf.matmul(conv11_RES,conv12_RES,transpose_b=True)
    output_1=Activation('softmax')(layer11)

    conv_2=Conv2D(filters=filter,kernel_size=(3,3),padding='same')(layer1)
    output_2=Reshape((conv_2.shape[1]*conv_2.shape[2],conv_2.shape[3]))(conv_2)

    conv_31=DepthwiseConv2D(kernel_size=(3,3))(layer1)
    conv_32=DepthwiseConv2D(kernel_size=(3,3))(layer1)
    conv32_RES=Reshape((conv_32.shape[1]*conv_32.shape[2],conv_32.shape[3]))(conv_32)
    conv31_RES=Reshape((conv_31.shape[1]*conv_31.shape[2],conv_31.shape[3]))(conv_31)
    layer12=tf.matmul(conv32_RES,conv31_RES,transpose_a=True)
    output_3=Activation('softmax')(layer12)

    output=tf.matmul(output_1,output_2)
    output=tf.matmul(output,output_3)
    layer1_1=Reshape((a,a,b))(output)



    layer1_2=Conv2D(filters=filter,kernel_size=(3,3),strides=1,padding='same')(layer1)
    layer1_3=layer1
    layer_output=tf.add(layer1_1,layer1_2)
    layer_output=tf.add(layer_output,layer1)
    layer_output=BatchNormalization()(layer_output)
    layer_output=Activation('relu')(layer_output)
    layer_output=MaxPooling2D(pool_size=(2,2),strides=2)(layer_output)
    return layer_output



attention_moudle(input_layer_Global,128,27,128)


def Local(input_layer):
    layer2=SE_Conv_moule_1(input_layer)
    layer3=SE_Conv_moule_2(layer2)
    layer4=SE_Conv_moule_2(layer3)
    layer5=SE_Conv_moule_3(layer4,64)
    output_layer=MaxPooling2D(padding='valid')(layer5)
    return output_layer,layer3



def Global(input_layer):
    layer2=attention_moudle(input_layer,128,27,128)
    layer4=SE_Res_module(layer2,128)
    layer5=SE_Res_module(layer4,128)
    layer6=MaxPooling2D(padding='valid')(layer5)
    layer7=SE_Conv_moule_3(layer6,64)
    layer8=MaxPooling2D(padding='valid')(layer7)
    layer9=SE_Conv_moule_3(layer8,64)
    output_layer=MaxPooling2D(strides=1,padding='same')(layer9)
    return output_layer,layer2



def MY_NET(input_layer):
    layer2=attention_moudle(input_layer,64,13,64)
    layer7=SE_Conv_moule_3(layer2,64)
    layer8=MaxPooling2D(padding='valid')(layer7)
    layer9=SE_Conv_moule_3(layer8,64)
    output_layer=MaxPooling2D(strides=1,padding='same')(layer9)
    return  output_layer


def MY_NET2(input_layer):
    layer2=attention_moudle(input_layer,64,7,64)
    layer5=SE_Conv_moule_3(layer2,64)
    output_layer=MaxPooling2D(strides=1,padding='same')(layer5)
    return output_layer



Local_W=7
Local_n_component=20
Global_W=27
Global_n_component=3


input_layer_local=Input((Local_W,Local_W,Local_n_component),name='input_layer_local') #input_layer_local=(7,7,20)
input_layer_Global=Input((Global_W,Global_W,Global_n_component),name='input_layer_Global')#input_layer_Glbal=(27,27,3)


output_layer_local,bb=Local(input_layer_local)    #Local 输出 (none,3,3,64)

output_layer_Global,b=Global(input_layer_Global) #Global 输出 (none,3,3,64)

c=MY_NET(b)
d=MY_NET2(bb)
print('c',c)

concat_layer=concatenate([output_layer_local,output_layer_Global,c,d],axis=-1)

flatten_layer=GlobalAveragePooling2D()(concat_layer)

Fully_connect_layer1=Dense(units=200,activation='sigmoid',kernel_regularizer=regularizers.l2(0.02))(flatten_layer)
Fully_conncet_layer2=Dense(units=100,activation='sigmoid')(Fully_connect_layer1)
output_layer_final=Dense(units=9,activation='softmax',name='output_layer_final')(Fully_conncet_layer2)


model=Model(inputs=[input_layer_local, input_layer_Global],outputs=output_layer_final)



