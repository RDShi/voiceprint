import sys
import os
import time
import random
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tflearn
import numpy as np
import argparse
import importlib
import threading
import utils


def get_config(filename):
    with open(filename, 'r') as f:
        f.readline()
        line = f.readline()
        par = line.strip().split(',')
        ratio_s = float(par[0])
        ratio_t = float(par[1])
        lr = float(par[2])
        disp_step = float(par[3])
        evaluate_step = float(par[4])

    return ratio_s, ratio_t, lr, disp_step, evaluate_step


def main(args):

    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), args.model_name)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), args.model_name)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    model_path = model_dir+'/'+args.model_name+'.ckpt'

    print('pid:'+str(os.getpid()))
    os.environ['CUDA_VISIBLE_DEVICES']= args.CUDA_VISIBLE_DEVICES

    # tf.reset_default_graph()

    network = importlib.import_module(args.model_def)

    max_checkpoints = 3

    lstm_model_setting={}
    lstm_model_setting['num_units']=128
    lstm_model_setting['dimension_projection']=args.embedding_size
    lstm_model_setting['attn_length']=10
    lstm_model_setting['num_layers']=3
    test_size = 10

    if args.pretrained_model and not args.finetuning:
        try:
            with open(os.path.join(os.path.dirname(args.pretrained_model),'test_speaker.txt'), 'r') as fid:
                test_speaker = fid.read().split('\n')
        except:
            test_speaker = random.sample(os.listdir(args.data_set), test_size)
    else:
        test_speaker = random.sample(os.listdir(args.data_set), test_size)

    with open(os.path.join(model_dir, 'test_speaker.txt'), 'w') as fid:
        fid.write('\n'.join(test_speaker))

    train_file = [os.path.join(args.data_set, file_name) for file_name in os.listdir(args.data_set) if file_name not in test_speaker]
    test_file = [os.path.join(args.data_set, file_name) for file_name in test_speaker]
    n_class = len(train_file)

    x = tf.placeholder(tf.float32, [None, 199, args.n_fbank], name='inputs')
    y = tf.placeholder(tf.int64, [None])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    lr_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    ratio_s_placeholder = tf.placeholder(tf.float32, name='softmax_rate')
    ratio_t_placeholder = tf.placeholder(tf.float32, name='triplet_rate')

    with tf.device('/cpu:0'):
        q = tf.FIFOQueue(args.batch_size*3, [tf.float32, tf.int64], shapes=[[199, args.n_fbank], []])
        enqueue_op = q.enqueue_many([x, y])
        x_b, y_b = q.dequeue_many(args.batch_size)

    # with tf.device('/gpu:0'):
    logits, embeddings = network.inference(x_b, lstm_model_setting, keep_prob, n_class)

    with tf.name_scope('loss'):
        with tf.name_scope('triplet_loss'):
            triplet_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(y_b, embeddings, margin=0.2)
            tf.summary.scalar('train', triplet_loss)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_b, logits=logits))
            tf.summary.scalar('train', softmax_loss)
        with tf.name_scope('total_loss'):
            total_loss = ratio_s_placeholder*softmax_loss + ratio_t_placeholder * triplet_loss
            tf.summary.scalar('train', total_loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), y_b), tf.float32))
        tf.summary.scalar('train', accuracy)

    # opt = tf.train.AdamOptimizer(learning_rate=lr_placeholder)
    opt = tf.train.MomentumOptimizer(learning_rate=lr_placeholder, momentum=0.9, use_nesterov=True)
    global_step = tf.Variable(0, trainable=False)
    gradients = opt.compute_gradients(total_loss)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    trainer = opt.apply_gradients(capped_gradients, global_step=global_step)
    # trainer = opt.minimize(total_loss, global_step=global_step)

    # merge ummaries to write them to file
    merged = tf.summary.merge_all()

    # checkpoint saver and restorer
    if args.only_weight:
        all_vars = tf.trainable_variables()
        excl_vars = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
        to_restore = [item for item in all_vars if tflearn.utils.check_restore_tensor(item, excl_vars)]
    elif args.finetuning:
        all_vars = tf.global_variables()
        excl_vars = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
        to_restore = [item for item in all_vars if tflearn.utils.check_restore_tensor(item, excl_vars)]
    else:
        to_restore = None

    restorer = tf.train.Saver(var_list=to_restore, max_to_keep=max_checkpoints)

    saver = tf.train.Saver(max_to_keep=max_checkpoints)

    coord = tf.train.Coordinator()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.pretrained_model:
            restorer.restore(sess, args.pretrained_model)

        # enqueuing batches procedure
        def enqueue_batches():
            while not coord.should_stop():
                batch_feats, batch_labels = utils.get_train_batch(train_file, args.batch_size)
                sess.run(enqueue_op, feed_dict={x: batch_feats, y: batch_labels})

        # creating and starting parallel threads to fill the queue
        num_threads = 3
        for i in range(num_threads):
            t = threading.Thread(target=enqueue_batches)
            t.setDaemon(True)
            t.start()

        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        start_time = time.time()

        step = sess.run(global_step)

        while step <= args.max_step:
            if args.config_file:
                ratio_s, ratio_t, lr, display_step, evaluate_step = get_config(args.config_file)
            else:
                ratio_s = 1
                ratio_t = 0.5
                lr = 0.001
                display_step = 100
                evaluate_step = 1000

            _, step = sess.run([trainer, global_step], feed_dict={keep_prob: args.keep_prob, lr_placeholder: lr, ratio_s_placeholder: ratio_s, ratio_t_placeholder: ratio_t})

            if step % display_step == 0:
                train_tl, train_sl, train_l, train_acc, result = sess.run([triplet_loss, softmax_loss, total_loss, accuracy, merged], 
                                                                          feed_dict={keep_prob: 1, lr_placeholder: lr, ratio_s_placeholder: ratio_s, ratio_t_placeholder: ratio_t})

                int_time = time.time()
                print('Step: {:09d} --- Loss: {:.7f} Cross Entropy: {:.07f} Triplet Loss: {:.07f} Training accuracy: {:.4f} Learning Rate: {} PID: {} Elapsed time: {}'
                      .format(step, train_l, train_sl, train_tl, train_acc, lr, os.getpid(), utils.format_time(int_time - start_time)))
                train_writer.add_summary(result, step)


            if step % evaluate_step == 0:
                enroll_feats, enroll_labels, test_feats, test_labels = utils.get_test_batch(test_file, args.batch_size)
                
                 
                embs = sess.run(embeddings, feed_dict={x_b: np.vstack((enroll_feats, test_feats[:args.batch_size-len(enroll_feats)])), keep_prob: 1})
                enroll_list = utils.enroll(embs[:len(enroll_feats)], enroll_labels)

                embsp = sess.run(embeddings, feed_dict={x_b: test_feats[args.batch_size-len(enroll_feats):], keep_prob: 1})
                predict_labels = utils.speaker_identification(np.vstack((embs[len(enroll_feats):],embsp)), enroll_list)
                test_acc = accuracy_score(test_labels, predict_labels)

                print('===================')
                int_time = time.time()
                print('Elapsed time: {}'.format(utils.format_time(int_time - start_time)))
                print('Validation accuracy: {:.04f}'.format(test_acc))
                # save weights to file
                save_path = saver.save(sess, model_path)
                print('Variables saved in file: %s' % save_path)
                print('Logs saved in dir: %s' % log_dir)
                summary = tf.Summary()
                summary.value.add(tag='accuracy/val', simple_value=test_acc)
                train_writer.add_summary(summary, step)
                print('===================')

        end_time = time.time()
        print ('Elapsed time: {}'.format(utils.format_time(end_time - start_time)))
        save_path = saver.save(sess, model_path)
        print('Variables saved in file: %s' % save_path)
        print('Logs saved in dir: %s' % log_dir)

        coord.request_stop()
        coord.join()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, help='pretrained model path.', default=None)
    parser.add_argument('--max_step', type=int, help='Number of steps to run.', default=1000000)
    parser.add_argument('--batch_size', type=int, help='batch size.', default=128)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, help='CUDA VISIBLE DEVICES', default='9')
    parser.add_argument('--config_file', type=str, help='File containing the learning rate schedule', default='./data/config.txt')
    parser.add_argument('--model_name', type=str, help='', default='test')
    parser.add_argument('--logs_base_dir', type=str, help='Directory where to write event logs.', default='~/logs/voice/VCTK/')
    parser.add_argument('--models_base_dir', type=str, help='Directory where to write trained models and checkpoints.', default='~/models/voice/VCTK/')
    parser.add_argument('--finetuning', type=bool, help='Whether finetuning.', default=False)
    parser.add_argument('--only_weight', type=bool, help="Whether only load pretrained model's weight.", default=False)
    parser.add_argument('--model_def', type=str, help='Model definition. Points to a module containing the definition of the inference graph.', default='models.lstm')
    parser.add_argument('--data_set', type=str, help="data set position.", default='/data/srd/data/VCTK-Corpus/wav48')
    parser.add_argument('--keep_prob', type=float, help="keep probability.", default=0.5)
    parser.add_argument('--n_fbank', type=int, help="the nunmber of filter bank oder.", default=40)
    parser.add_argument('--embedding_size', type=int, help="embedding size.", default=256)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


