import numpy
import tensorflow
import json
from sklearn.model_selection import train_test_split


FILE_PATH = ####
INPUT_TRAIN_DATA_FILE_NAME = #.npy file#
LABEL_TRAIN_DATA_FILE_NAME = #.npy file#
DATA_CONFIGS_FILE_NAME = #json file#

input_data = np.load(open(FILE_DIR_PATH + INPUT_TRAIN_DATA_FILE_NAME, 'rb'))
label_data = np.load(open(FILE_DIR_PATH + LABEL_TRAIN_DATA_FILE_NAME, 'rb'))
prepro_configs = json.load(open(FILE_DIR_PATH + DATA_CONFIGS_FILE_NAME, 'r'))

TEST_SPLIT = 0.1
RNG_SEED = 13371447

# split train, evaluation data
input_train, input_eval, label_train, label_eval = train_test_split(input_data, label_data, test_size=TEST_SPLIT, random_state=RNG_SEED)

def mapping_fn(X, Y):
    input, label = {'text': X}, Y
    return input, label

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
