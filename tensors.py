import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import activations, optimizers, losses
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import io

data = pd.read_csv("/Users/danielsaggau/PycharmProjects/pythonProject/data/working_sentence_split.csv")
MODEL_NAME = 'distilbert-base-uncased'
tkzr = DistilBertTokenizer.from_pretrained(MODEL_NAME)
def construct_encodings(x, tkzr, max_len, trucation=True, padding=True):
    return tkzr(x, max_length=max_len, truncation=trucation, padding=padding)


MAX_LEN = 20

x = list(data['beginning'])
y = list(data['true_end'])
encodings = construct_encodings(data['beginning'], tkzr, max_len=MAX_LEN)

def construct_tfdataset(encodings, y=None):
    if y:
        return tf.data.Dataset.from_tensor_slices((dict(encodings), y))
    else:
        # this case is used when making predictions on unseen samples after training
        return tf.data.Dataset.from_tensor_slices(dict(encodings))


tfdataset = construct_tfdataset(encodings, y)

TEST_SPLIT = 0.2
BATCH_SIZE = 2

train_size = int(len(x) * (1-TEST_SPLIT))

tfdataset = tfdataset.shuffle(len(x))
tfdataset_train = tfdataset.take(train_size)
tfdataset_test = tfdataset.skip(train_size)

tfdataset_train = tfdataset_train.batch(BATCH_SIZE)
tfdataset_test = tfdataset_test.batch(BATCH_SIZE)

