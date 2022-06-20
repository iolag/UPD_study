def train(self, dataset):
    # Determine trainable variables
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # Build losses
    self.losses['L1'] = tf.losses.absolute_difference(
        self.x, self.reconstruction, reduction=Reduction.NONE)
    rec = tf.reduce_sum(self.losses['L1'], axis=[1, 2, 3])
    kl = 0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.square(self.z_sigma) -
                             tf.log(tf.square(self.z_sigma)) - 1, axis=1)
    self.losses['pixel_loss'] = rec + kl
    self.losses['reconstructionLoss'] = tf.reduce_mean(rec)
    self.losses['kl'] = tf.reduce_mean(kl)
    self.losses['loss'] = tf.reduce_mean(rec + kl)

    # for restoration
    self.losses['restore'] = self.tv_lambda * tf.image.total_variation(
        tf.subtract(self.x, self.reconstruction))
    self.losses['grads'] = tf.gradients(self.losses['pixel_loss'] + self.losses['restore'], self.x)[0]


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:26:16 2018

Restoration on VAE

@author: syou
"""

import numpy as np
import tensorflow as tf
import os
import pandas as pd
from vars_layers import input_set
from vae_graph import q_zx, p_xz, loss
from utils import num2str, path_make, data_load
from datetime import datetime
import h5py


# Parameter Settings


tv_lambda = np.array([2.2])
batchsize = 60  # restoration batch
imageshape = [158, 198]
dim_z = 1  # latent variable z's channel
dim_x = imageshape[0] * imageshape[1]  # dimension of input
clipstd = [0.0, 1.0]

restore_steps = 500
gradient_clip = True
clipbound = 50

# input

tf.reset_default_graph()
x, x_reshape = input_set(dim_x, imageshape[0], imageshape[1])
x_p, x_p_reshape = input_set(dim_x, imageshape[0], imageshape[1])
bmask_in = tf.placeholder(shape=[None] + imageshape, dtype=tf.float32)
bmask_reshape = tf.reshape(bmask_in, shape=[-1, dim_x])
tv_lamda = tf.placeholder(tf.float32, shape=())

# the same graph applied

z_sampled, z_mean, z_std = q_zx(x)

xz_mean, xz_logvarinv = p_xz(z_sampled)
_, z_loss, _ = loss(z_mean, z_std, xz_mean, xz_logvarinv, x_reshape)
l2 = tf.reduce_sum(tf.squared_difference(tf.reshape(xz_mean, [-1, dim_x]), x), axis=1)
xz_std = tf.exp(- 0.5 * xz_logvarinv)
# the loss for the restoration


loss1 = -1 * (l2 + z_loss)
loss2 = - tv_lamda * tf.image.total_variation((input - restored))

# Gradient dloss/dy
grads = tf.gradients([loss1, loss2], [x])[0]
upperbound = tf.cast(tf.fill(tf.shape(grads), clipbound), dtype=tf.float32)
lowerbound = tf.cast(tf.fill(tf.shape(grads), -1 * clipbound), dtype=tf.float32)
clipgrads = tf.clip_by_value(grads, lowerbound, upperbound, name='cliped_updating_gradient')
gradimage = tf.reshape(clipgrads, shape=tf.stack([-1] + imageshape))
gradsmasked = tf.multiply(clipgrads, bmask_reshape)

#


for k in np.arange(batch + 1):

    bmask = 1  # mask
    rawData = 1  # slice
    labels = MRlabel[list(index)][:, 22:180, 17:215]
    restored = rawData.reshape(-1, dim_x)  # flattens
    input = restored.copy()
    lr = 1e-3

    for step in range(restore_steps):

        restored += lr * sess.run(gradsmasked,
                                  feed_dict={x: restored, x_p: input, tv_lamda: tv_lamda, bmask_in: bmask})
