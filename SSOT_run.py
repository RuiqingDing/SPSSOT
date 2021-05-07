# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pylab as pl
import random
import matplotlib.pyplot as plt
from config import *
from utils.metrics import val_model_binary
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Lambda
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from models.SSOT import DeepSemiOT

flags =  tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('source', 'mimic', 'name of the source data')
flags.DEFINE_string('target', 'challenge', 'name of the target data')
flags.DEFINE_string('percent', '1-79', 'target labeled data percentage')
flags.DEFINE_integer('batch_size', 256, 'the batch size of training')
flags.DEFINE_float('int_lr', 0.001, "Learning rate of for SSOT")
flags.DEFINE_float('sloss', 1.0, 'weight of the cross entropy of source data')
flags.DEFINE_float('tloss', 2.0, 'weight of the cross entropy of target labeled data')
flags.DEFINE_float('gloss', 1.0, 'the weight of group entropy of target unlabeled data')
flags.DEFINE_float('ot_alpha', 0.05, 'ot weight')
flags.DEFINE_float('closs', 0.1, 'the weight of center distance')
flags.DEFINE_float('source_percent', 1.0, 'the percentage of source domain')

Xs, ys, Xtl, ytl, Xtu, ytu = load_data(source=FLAGS.source, target=FLAGS.target, percent=FLAGS.percent, source_percent=FLAGS.source_percent)
Xtest, ytest = load_test_data(target=FLAGS.target)

# convert to one hot encoded vector
ys_cat = to_categorical(ys)
ytl_cat = to_categorical(ytl)
ytu_cat = to_categorical(ytu)
ytest_cat = to_categorical(ytest)

# optimizer
n_class = len(np.unique(ys))
n_dim = np.shape(Xs)
optim = SGD(lr=0.001)

# %% feature extraction and classifier function definition
def feat_ext(main_input, l2_weight=0.0):
    net = Dense(256, activation='relu', name='fe')(main_input)
    net = Dense(128, activation='relu', name='feat_ext')(net)
    return net

def classifier(model_input, nclass, l2_weight=0.0):
    net = Dense(128, activation='relu', name='cl')(model_input)
    net = Dense(nclass, activation='softmax', name='cl_output')(net)
    return net

# Feature extraction as a keras model
main_input = Input(shape=(n_dim[1],))
fe = feat_ext(main_input)
fe_size = fe.get_shape().as_list()[1]
# feature extraction model
fe_model = Model(main_input, fe, name='fe_model')
# Classifier model as a keras model
cl_input = Input(shape=(fe.get_shape().as_list()[1],))  # input dim for the classifier
net = classifier(cl_input, n_class)
# classifier keras model
cl_model = Model(cl_input, net, name='classifier')

# model train with
ms = Input(shape=(n_dim[1],))
fes = feat_ext(ms)
nets = classifier(fes, n_class)
source_model = Model(ms, nets)
source_model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
source_model.fit(Xs, ys_cat, batch_size=128, epochs=50, validation_data=(Xtl, ytl_cat), verbose = 0)

# Finetune on target labeled data
source_model.fit(Xtl, ytl_cat, batch_size=128, epochs=50, validation_data=(Xtest, ytest_cat), verbose = 0)

# Target model
print('\n')
main_input = Input(shape=(n_dim[1],))
# feature extraction model
ffe=fe_model(main_input)
# classifier model
net = cl_model(ffe)
# target model with two outputs: predicted class prob, and intermediate layers
model = Model(inputs=main_input, outputs=[net, ffe])
model.set_weights(source_model.get_weights())


# DeepSemiOT model initalization
al_model = DeepSemiOT(model, FLAGS.batch_size, n_class, optim, label_percent=0.5, sloss = FLAGS.sloss, tloss = FLAGS.tloss, gloss = FLAGS.gloss, closs=FLAGS.closs, int_lr = FLAGS.int_lr, ot_alpha= FLAGS.ot_alpha, lr_decay=True, verbose=1,ot_method='emd')
# DeepSemiOT model fit
al_model.fit(Xs, ys_cat, Xtl, ytl_cat, Xtu, Xtest, ytest_cat, target_label=None, n_iter=5000, cal_bal=False)

probs_t = al_model.evaluate(Xtest)
print("\n test loss & acc using SSOT model")
val_model_binary(ytest, probs_t)