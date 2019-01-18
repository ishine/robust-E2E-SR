# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:37:00 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:30:16 2018

@tsx: Administrator

based learining noise-invariant representations for robust speech recognition
towards end2end speech recognition baesd cnn
"""




import tensorflow as tf
import numpy as np
import os
import random
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


# In[25]:


def get_weight_variable(shape, name):
    w = tf.get_variable(name,
                        shape,
                        initializer=tf.random_uniform_initializer(-0.05, 0.05))
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(1e-5)(w))
    return w
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n ] *len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1 ] +1], dtype=np.int64)
    return indices, values, shape
def shuffle_every_epoch(DIR):
    alist = os.listdir(DIR)
    random.shuffle(alist)
    return alist
def batch_list(DIR, batch_size):
    alist = shuffle_every_epoch(DIR)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(alist):
            batch_count = 0
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield alist[start:end]



def conv2d(x, W, b):
    return tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 2, 1], padding='SAME'), b)
def conv2d_1(x, W, b):
    return tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'), b)
def deconv2d(x, W, b, output_shape):
    return tf.nn.bias_add(tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 2, 1], padding='SAME'), b)
def deconv2d_1(x, W, b, output_shape):
    return tf.nn.bias_add(tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 1, 1], padding='SAME'), b)
def batch_training_noise(DIR, batch_list):
    clean_batch = []
    noise_batch = []
    seq_len = []
    index = []
    for file in batch_list:
        npz_file = np.load(os.path.join(DIR, file))
        clean_batch.append(npz_file['clean'])
        noise_batch.append(npz_file['noise'])
        seq_len.append(npz_file['frames'])
        index.append(npz_file['index'])
    maxlen = max(seq_len)
    for i in range(len(batch_list)):
        clean_batch[i] = np.pad(clean_batch[i], ((0, maxlen - seq_len[i]), (0, 0)), 'constant')
        noise_batch[i] = np.pad(noise_batch[i], ((0, maxlen - seq_len[i]), (0, 0)), 'constant')
    clean = np.expand_dims(np.array(clean_batch), axis=3)
    noise = np.expand_dims(np.array(noise_batch), axis=3)
    sparse_index = sparse_tuple_from(index)
    return clean, noise,maxlen, seq_len, sparse_index
def batch_training(DIR, batch_list):
    clean_batch = []
#     noise_batch = []
    seq_len = []
    index = []
    for file in batch_list:
        npz_file = np.load(os.path.join(DIR, file))
        clean_batch.append(npz_file['clean'])
#         noise_batch.append(npz_file['noise'])
        seq_len.append(npz_file['frames'])
        index.append(npz_file['index'])
    maxlen = max(seq_len)
    for i in range(len(batch_list)):
        clean_batch[i] = np.pad(clean_batch[i], ((0, maxlen - seq_len[i]), (0, 0)), 'constant')
#         noise_batch[i] = np.pad(noise_batch[i], ((0, maxlen - seq_len[i]), (0, 0)), 'constant')
    clean = np.expand_dims(np.array(clean_batch), axis=3)
#     noise = np.expand_dims(np.array(noise_batch), axis=3)
    sparse_index = sparse_tuple_from(index)
    return clean,maxlen, seq_len, sparse_index

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# In[86]:
s = time.time()
Batch_size = 4
ckpt_dir = './experiment10/ckpt'
model_name = 'experiment10'
LR2 = 1e-5
rate=0.3
g1 = tf.Graph()
with g1.as_default():
    with tf.variable_scope('placeholder'):
        clean = tf.placeholder(tf.float32, name='clean')
        noise = tf.placeholder(tf.float32, name='noise')
        seq_len = tf.placeholder(tf.int32, name='seq_len')
        frames = tf.placeholder(tf.int32, name='frames')
        batch_size = tf.placeholder(tf.int32, name='batch_size')
        labels = tf.sparse_placeholder(tf.int32, name='labels')
        training = tf.placeholder(tf.bool, name='dropout_if')
    
    
        
    with tf.variable_scope('recognition1'):
        re_w1 = get_weight_variable(name='re_w1', shape=[5, 3, 1,128] )
        re_b1 = get_weight_variable(name='re_b1', shape=[128])
        z_c_1 = tf.layers.dropout(tf.nn.relu(conv2d(clean, re_w1, re_b1)),rate=rate,training=training)
    with tf.variable_scope('recognition1', reuse=True):
        re_w1 = get_weight_variable(name='re_w1', shape=[5, 3, 1,128] )
        re_b1 = get_weight_variable(name='re_b1', shape=[128])
        z_n_1 = tf.layers.dropout(tf.nn.relu(conv2d(noise, re_w1, re_b1)),rate=rate,training=training)
    mse1 = tf.reduce_sum((z_n_1 - z_c_1) ** 2, axis=(1, 2, 3))
        
    
    with tf.variable_scope('recognition2'):
        re_w2 = get_weight_variable(name='re_w2', shape=[5, 3, 128, 128] )
        re_b2 = get_weight_variable(name='re_b2', shape=[128])
        z_c_2 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_c_1, re_w2, re_b2)), rate=rate,training=training)
    with tf.variable_scope('recognition2', reuse=True):
        re_w2 = get_weight_variable(name='re_w2', shape=[5, 3, 128, 128] )
        re_b2 = get_weight_variable(name='re_b2', shape=[128])
        z_n_2 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_n_1, re_w2, re_b2)), rate=rate,training=training)
    mse2 = tf.reduce_sum((z_n_2 - z_c_2) ** 2, axis=(1, 2, 3))    
#    
        
    with tf.variable_scope('recognition3'):    
        re_w3 = get_weight_variable(name='re_w3', shape=[5, 3, 128, 128] )
        re_b3 = get_weight_variable(name='re_b3', shape=[128])
        z_c_3 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_c_2, re_w3, re_b3)),rate=rate,training=training)
    with tf.variable_scope('recognition3', reuse=True):    
        re_w3 = get_weight_variable(name='re_w3', shape=[5, 3, 128, 128] )
        re_b3 = get_weight_variable(name='re_b3', shape=[128])
        z_n_3 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_n_2, re_w3, re_b3)),rate=rate,training=training)
    mse3 = tf.reduce_sum((z_n_3 - z_c_3) ** 2, axis=(1, 2, 3))
        
    with tf.variable_scope('recognition4'):
        re_w4 = get_weight_variable(name='re_w4', shape=[5, 3, 128, 128] )
        re_b4 = get_weight_variable(name='re_b4', shape=[128])
        z_c_4 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_c_3, re_w4, re_b4)),rate=rate,training=training)
    with tf.variable_scope('recognition4', reuse=True):
        re_w4 = get_weight_variable(name='re_w4', shape=[5, 3, 128, 128] )
        re_b4 = get_weight_variable(name='re_b4', shape=[128])
        z_n_4 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_n_3, re_w4, re_b4)),rate=rate,training=training)
    mse4 = tf.reduce_sum((z_n_4 - z_c_4) ** 2, axis=(1, 2, 3))
        
    with tf.variable_scope('recognition5'):
        re_w5 = get_weight_variable(name='re_w5', shape=[5, 3,128, 128] )
        re_b5 = get_weight_variable(name='re_b5', shape=[128])
        z_c_5 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_c_4, re_w5, re_b5)),rate=rate,training=training)
    with tf.variable_scope('recognition5', reuse=True):
        re_w5 = get_weight_variable(name='re_w5', shape=[5, 3,128, 128] )
        re_b5 = get_weight_variable(name='re_b5', shape=[128])
        z_n_5 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_n_4, re_w5, re_b5)),rate=rate,training=training)
    mse5 = tf.reduce_sum((z_n_5 - z_c_5) ** 2, axis=(1, 2, 3))
        
    with tf.variable_scope('recognition6'):
        re_w6= get_weight_variable(name='re_w6', shape=[5, 3, 128, 128])
        re_b6 = get_weight_variable(name='re_b6', shape=[128])
        z_c_6 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_c_5, re_w6, re_b6)),rate=rate, training=training)
    with tf.variable_scope('recognition6', reuse=True):
        re_w6= get_weight_variable(name='re_w6', shape=[5, 3, 128, 128])
        re_b6 = get_weight_variable(name='re_b6', shape=[128])
        z_n_6 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_n_5, re_w6, re_b6)),rate=rate, training=training)
    mse6 = tf.reduce_sum((z_n_6 - z_c_6) ** 2, axis=(1, 2, 3))
        
    with tf.variable_scope('recognition7'):
        re_w7= get_weight_variable(name='re_w7', shape=[5, 3, 128, 128])
        re_b7 = get_weight_variable(name='re_b7', shape=[128])
        z_c_7 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_c_6, re_w7, re_b7)),rate=rate, training=training)
    with tf.variable_scope('recognition7', reuse=True):
        re_w7 = get_weight_variable(name='re_w7', shape=[5, 3, 128, 128])
        re_b7 = get_weight_variable(name='re_b7', shape=[128])
        z_n_7 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_n_6, re_w7, re_b7)),rate=rate, training=training)
    mse7 = tf.reduce_sum((z_n_7 - z_c_7) ** 2, axis=(1, 2, 3))
        
#    with tf.variable_scope('recognition8'):
#        re_w8= get_weight_variable(name='re_w8', shape=[5, 3, 128, 128])
#        re_b8 = get_weight_variable(name='re_b8', shape=[128])
#        z_c_8 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_c_7, re_w8, re_b8)),rate=rate, training=training)
#    with tf.variable_scope('recognition8', reuse=True):
#        re_w8 = get_weight_variable(name='re_w8', shape=[5, 3, 128, 128])
#        re_b8 = get_weight_variable(name='re_b8', shape=[128])
#        z_n_8 = tf.layers.dropout(tf.nn.relu(conv2d_1(z_n_7, re_w8, re_b8)),rate=rate, training=training)

    
    with tf.variable_scope('re_fc_1'):
        re_fc_w1 = get_weight_variable(name='re_fc_w1', shape=[32*128, 512] )
        re_fc_b1 = get_weight_variable(name='re_fc_b1', shape=[512])
        fc_c_1 = tf.layers.dropout(tf.nn.relu(tf.matmul(tf.reshape(z_c_7, [batch_size*frames, -1]), re_fc_w1) + re_fc_b1),rate=rate,training=training)
    with tf.variable_scope('re_fc_1', reuse=True):
        re_fc_w1 = get_weight_variable(name='re_fc_w1', shape=[32*128, 512] )
        re_fc_b1 = get_weight_variable(name='re_fc_b1', shape=[512])
        fc_n_1 = tf.layers.dropout(tf.nn.relu(tf.matmul(tf.reshape(z_n_7, [batch_size*frames, -1]), re_fc_w1) + re_fc_b1),rate=rate,training=training)
    mse7 = tf.reduce_sum((tf.reshape(fc_n_1, [batch_size, frames, -1]) - tf.reshape(fc_c_1, [batch_size, frames, -1])) ** 2, axis=(1, 2)) 
    with tf.variable_scope('re_fc_2'):     
        re_fc_w2 = get_weight_variable(name='re_fc_w2', shape=[512, 512] )
        re_fc_b2 = get_weight_variable(name='re_fc_b2', shape=[512])
        fc_c_2 = tf.layers.dropout(tf.nn.relu(tf.matmul(fc_c_1, re_fc_w2) + re_fc_b2),rate=rate,training=training)
    with tf.variable_scope('re_fc_2', reuse=True):     
        re_fc_w2 = get_weight_variable(name='re_fc_w2', shape=[512, 512] )
        re_fc_b2 = get_weight_variable(name='re_fc_b2', shape=[512])
        fc_n_2 = tf.layers.dropout(tf.nn.relu(tf.matmul(fc_n_1, re_fc_w2) + re_fc_b2),rate=rate, training=training)
    mse8 = tf.reduce_sum((tf.reshape(fc_n_2, [batch_size, frames, -1]) - tf.reshape(fc_c_2, [batch_size, frames, -1])) ** 2, axis=(1, 2)) 
    
    
    with tf.variable_scope('ctc_loss'):
        re_logit_w = get_weight_variable(name='re_logit_w', shape=[512, 62] )
        re_logit_b = get_weight_variable(name='re_logit_b', shape=[62])
        logit_c =tf.reshape(tf.matmul(fc_c_2, re_logit_w) + re_logit_b, [batch_size, frames, 62])
    with tf.variable_scope('ctc_loss', reuse=True):
        re_logit_w = get_weight_variable(name='re_logit_w', shape=[512, 62] )
        re_logit_b = get_weight_variable(name='re_logit_b', shape=[62])
        logit_n =tf.reshape(tf.matmul(fc_n_2, re_logit_w) + re_logit_b, [batch_size, frames, 62])
    logit = tf.reduce_mean(tf.reduce_sum((logit_n - logit_c)**2, axis=(1,2)))
        
    with tf.variable_scope('cosin_loss'):
         x_c_norm = tf.sqrt(tf.reduce_sum(tf.square(logit_c), axis=(2)))
         x_n_norm =  tf.sqrt(tf.reduce_sum(tf.square(logit_n), axis=(2)))
         xc_xn = tf.reduce_sum(tf.multiply(logit_c, logit_n), axis=(2))
         cosin = tf.reduce_mean(xc_xn/tf.multiply(x_c_norm, x_n_norm))
         
        
    with tf.variable_scope('loss_c'):
        cost_c = tf.nn.ctc_loss(labels=labels, inputs=logit_c, sequence_length=seq_len, time_major=False)
        ctc_c = tf.reduce_mean(cost_c)
    with tf.variable_scope('loss_n'):
        cost_n = tf.nn.ctc_loss(labels=labels, inputs=logit_n, sequence_length=seq_len, time_major=False)
        ctc_n = tf.reduce_mean(cost_n)
  
    loss_3 = ctc_n + ctc_c +1e-4*logit 
    tf.add_to_collection('loss', loss_3)
    loss = tf.add_n(tf.get_collection('loss'))
    optimizer = tf.train.AdamOptimizer(learning_rate=LR2).minimize(loss_3)
    
    with tf.variable_scope('LER_c'):
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(logit_c, [1, 0, 2]), seq_len,10)
        ler_c = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels, True))
    with tf.variable_scope('LER_n'):
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(logit_n, [1, 0, 2]), seq_len,10)
        ler_n = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels, True))
        

with tf.Session(graph=g1, config=config) as sess:
    
    tf.global_variables_initializer().run()

    saver=tf.train.Saver(max_to_keep=300)
    saver.restore(sess,'./experiment10/ckpt/experiment10-80')
#
    for epoch in range(81,100):
        List = batch_list('./tr+10+5', Batch_size)
        for i in range(1, int(7300/Batch_size)):
            alist = next(List)
            clean_batch,noise_batch, maxlen, seq, sparse_index = batch_training_noise('./tr+10+5', alist)
            feed_dict = {clean: clean_batch,
                         noise: noise_batch,
                         frames: maxlen,
                         batch_size: Batch_size,
                         labels:sparse_index,
                         seq_len:seq,
                         training:True}
            Ctc_c,Ctc_n,Cosin,Logit,Ler_c, Ler_n , _ = sess.run([ctc_c, ctc_n, cosin, logit, ler_c, ler_n, optimizer], feed_dict=feed_dict)
            if i % 100 == 0:
                 print('ctc_c:%f, ctc_n:%f, cosin:%f, logit:%f, ler_c:%f, ler_n:%f,i:%d, epoch:%f'%(Ctc_c,Ctc_n,Cosin,Logit,Ler_c, Ler_n,i,epoch))
        saver.save(sess, os.path.join(ckpt_dir, model_name), global_step=epoch)
        if epoch % 1 ==0:
            L = 0
            List_test = batch_list('./test_set', 1)
            for i in range(192):
                 List_t = next(List_test)
                 clean_batch, maxlen, seq, sparse_index = batch_training('./test_set', List_t)
                 feed_dict={
                         clean: clean_batch,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 clean:clean_batch,
                         frames: maxlen,
                         seq_len:seq,
                         batch_size: 1,
                         labels:sparse_index,
                         training:False}
                 Ler = sess.run(ler_c, feed_dict=feed_dict)
                 L += Ler
            print('clean_set', L/192)
#            
#            List_test_05 = batch_list('./hfchannel+00', 1)
#            L = 0
#            for i in range(192):
#                 L_05 = next(List_test_05)
#                 clean_batch, noise_batch, maxlen, seq, sparse_index = batch_training_noise('./hfchannel+00', L_05)
#                 feed_dict = {noise: noise_batch,
#                              clean:clean_batch,
#                         frames: maxlen,
#                         seq_len:seq,
#                         batch_size: 1,
#                         labels:sparse_index,
#                         training:False}
#                 Ler = sess.run(ler_n, feed_dict=feed_dict)
#                 L += Ler
#            print('hfchannel+00', L/192)
#            
#            List_test_05 = batch_list('./hfchannel+05', 1)
#            L = 0
#            for i in range(192):
#                 L_05 = next(List_test_05)
#                 clean_batch, noise_batch, maxlen, seq, sparse_index = batch_training_noise('./hfchannel+05', L_05)
#                 feed_dict = {noise: noise_batch,
#                              clean:clean_batch,
#                         frames: maxlen,
#                         seq_len:seq,
#                         batch_size: 1,
#                         labels:sparse_index,
#                         training:False}
#                 Ler = sess.run(ler_n, feed_dict=feed_dict)
#                 L += Ler
#            print('hfchannel+05', L/192)
#            
#            List_test_05 = batch_list('./hfchannel+10', 1)
#            L = 0
#            for i in range(192):
#                 L_05 = next(List_test_05)
#                 clean_batch, noise_batch, maxlen, seq, sparse_index = batch_training_noise('./hfchannel+10', L_05)
#                 feed_dict = {noise: noise_batch,
#                              clean:clean_batch,
#                         frames: maxlen,
#                         seq_len:seq,
#                         batch_size: 1,
#                         labels:sparse_index,
#                         training:False}
#                 Ler = sess.run(ler_n, feed_dict=feed_dict)
#                 L += Ler
#            print('snr_test_05', L/192)
#            List_test_05 = batch_list('./log-05', 1)
#            L = 0
#            for i in range(182):
#                 L_05 = next(List_test_05)
#                 clean_batch, noise_batch, maxlen, seq, sparse_index = batch_training_noise('./log-05', L_05)
#                 feed_dict = {noise: noise_batch,
#                              clean:clean_batch,
#                         frames: maxlen,
#                         seq_len:seq,
#                         batch_size: 1,
#                         labels:sparse_index,
#                         training:False}
#                 Ler = sess.run(ler_n, feed_dict=feed_dict)
#                 L += Ler
#            print('test-05', L/182)
                    
#            
#              
