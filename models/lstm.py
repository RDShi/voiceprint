import tensorflow as tf
import tflearn


def inference(incoming, lstm_model_setting, keep_prob, n_class, reuse=tf.AUTO_REUSE, finetuning=False):
    
    with tf.variable_scope('LSTM', reuse=reuse):
        incoming = tf.unstack(incoming, incoming.get_shape().as_list()[1], 1)

        lstm_cell = tf.contrib.rnn.LSTMCell(num_units=lstm_model_setting['num_units'], num_proj=lstm_model_setting['dimension_projection'])
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        lstm_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_cell,attn_length=lstm_model_setting['attn_length']) #add att

        outputs, _ = tf.nn.static_rnn(lstm_cell, incoming, dtype=tf.float32)
        logits = tflearn.fully_connected(outputs[-1], n_class, restore=not finetuning)
        embeddings = tf.nn.l2_normalize(outputs[-1], axis=1)

    return logits, embeddings


