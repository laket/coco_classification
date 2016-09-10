#!/usr/bin/python

import os
import time
import numpy as np
import tensorflow as tf
import coco_input
import model
import config

FLAGS = tf.app.flags.FLAGS


def get_opt(loss, global_step):
    lr = tf.train.exponential_decay(FLAGS.lr,
                                    global_step,
                                    FLAGS.decay_steps,
                                    FLAGS.decay_rate,
                                    staircase=True)
    
    opt = tf.train.MomentumOptimizer(lr, momentum=0.95)
    opt_op = opt.minimize(loss, global_step=global_step)

    tf.scalar_summary("lr", lr)

    return lr, opt_op

def train():
    global_step = tf.Variable(0, trainable=False)
    
    labels, images = coco_input.validate_input()

    network = model.Network(is_train=True)
    logits = network.inference(images)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    
    entropy, loss = model.get_loss(labels, logits)
    lr, opt = get_opt(loss, global_step)

    saver = tf.train.Saver(tf.trainable_variables())
    summary_op = tf.merge_all_summaries()
    
    gpu_options = tf.GPUOptions(allow_growth=True) 
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        summary_writer = tf.train.SummaryWriter("log", sess.graph)        
        
        tf.train.start_queue_runners(sess=sess)

        for num_iter in range(1,FLAGS.max_steps+1):
            start_time = time.time()
            value_entropy, value_loss, value_lr, _ = sess.run([entropy, loss, lr, opt])
            duration = time.time() - start_time
            assert not np.isnan(value_loss), 'Model diverged with loss = NaN'

            if num_iter % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                print ("entropy = {:.2f} loss = {:.2f} ({:.1f} examples/sec; {:.1f} sec/batch)"
                       .format(value_entropy, value_loss, examples_per_sec, sec_per_batch))

            if num_iter % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, num_iter)
                
            if num_iter % 1000 == 0:
                print "lr = {:.2f}".format(value_lr)
                checkpoint_path = os.path.join(FLAGS.dir_parameter, 'model.ckpt')                
                saver.save(sess, checkpoint_path,global_step=num_iter)
                

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.dir_log):
        tf.gfile.DeleteRecursively(FLAGS.dir_log)
    tf.gfile.MakeDirs(FLAGS.dir_log)

    if tf.gfile.Exists(FLAGS.dir_parameter):
        tf.gfile.DeleteRecursively(FLAGS.dir_parameter)
    tf.gfile.MakeDirs(FLAGS.dir_parameter)
    
    train()

if __name__ == '__main__':
    tf.app.run()

    