import tensorflow as tf
import numpy as np

import data_fn

BATCH_SIZE = #size of batch#
NUM_CLASS = #number of class#
NUM_EPOCHS = #number of epochs#
EMBED_DIM = #word embedding dimension# #


def loadGloVe(filename):

    embedding = {}
    vocab_size = 0

    file = open(filename,'r', encoding ='utf-8')
    print('loading glove vector')

    for line in file.readlines():
        row = line.strip().split(' ')
        embedd_dict[row[0]] = np.asarray(row[1:], dtype='float32')
        vocab_size = vocab_size + 1
    file.close()

    return ebedding, vocab_size

embedding, vocab_size = loadGloVe(PRETRAIN_PATH + PRETRAIN_FILE)

embedding_matrix = np.random.uniform( -1, 1, size = (vocab_size + 1, embed_dim)) # UNK은 random값
embedding_matrix[0] = np.zeros(embed_dim) # padding

for w, i in data_configs.items():

    vector = embedding[w]

    if vector in not None and i <= vocab_size
        embedding_matrix[i] = vector

def initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix

def model_fn(features, labels, mode, params):

    TRAIN = (mode = tf.estimator.ModeKeys.TRAIN)
    EVAL = (mode = tf.estimator.ModeKeys.EVAL)
    PREDICT = (mode = tf.estimator.ModeKeys.PREDICT)

    input_layer = tf.conrib.layer.embed_sequence(
                    ids = features['text'],
                    vocab_size = vocab_size,
                    embed_dim  = EMBED_DIM,
                    trainable = True, # option
                    initializer = params['embedding_initializer'],
                    )

    conv_3 = tf.layers.con1d(
            input = input_layer,
            filter = 100,
            kernel_size = 3,
            padding='same',
            use_bias=True,
            activation=tf.nn.relu
            )

    pooled_3 = tf.reduce_max(input_tensor=conv_3, axis=1)

    conv_5 = tf.layers.con1d(
            input = input_layer, #앞 drop할지안할지 결정필요
            filter = 100,
            kernel_size = 3,
            padding='same',
            use_bias=True,
            activation=tf.nn.relu
            )

    pooled_5 = tf.reduce_max(input_tensor=conv_5, axis=1)

    conv_7 = tf.layers.con1d(
            input = input_layer, #앞 drop할지안할지 결정필요
            filter = 100,
            kernel_size = 7,
            padding='same',
            use_bias=True,
            activation=tf.nn.relu
            )

    pooled_7 = tf.reduce_max(input_tensor=conv_7, axis=1)

    concat = tf.concat(
            [pooled_3, pooled_5, pooled_7],
            axis = -1,
            name = 'concat'
            )

    concat_flat = tf.layers.flatten( input = concat, name='concat_flat')
    regularizer = tf.contrib.layers.l2_regularizer(scale=3)
    dropout_hidden = tf.layers.dropout(inputs=concat_flat, rate=0.5, training=TRAIN)
    logits = tf.layers.dense(
                inputs=dropout_hidden,
                units=NUM_CLASS,
                regularizer = regularizer,
                kernel_initializer = tf.contrib.layers.xavier_initializer())

    if TRAIN:
        global_step = tf.train.get_global_step()
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss = loss)

    if EVAL:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        pred = tf.argmax(logits, 1, name="predictions")
        accuracy = tf.metrics.accuracy(labels, pred)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})

    if TEST:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        pred = tf.argmax(logits, 1, name="predictions")
        accuracy = tf.metrics.accuracy(labels, pred)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})

params = {'embedding_initializer': initializer}

model_dir = os.path.join(os.getcwd(), "checkpoint/cnn_model")
os.makedirs(model_dir, exist_ok=True)

config_tf = tf.estimator.RunConfig()
config_tf._save_checkpoints_steps = 100
config_tf._save_checkpoints_secs = None
config_tf._keep_checkpoint_max =  2
config_tf._log_step_count_steps = 100

est = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config_tf, params=params)

print("             train process              ")
print("........................................")

tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)
time_start = datetime.utcnow()
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................")

est.train(data_fn.train_input_fn)

time_end = datetime.utcnow()
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))

print("             eval process              ")
print(".......................................")

valid = est.evaluate(data_fn.eval_input_fn)

print("             test process              ")
print(".......................................")

test = est.evaluate(data_fn.test_input_fn)
