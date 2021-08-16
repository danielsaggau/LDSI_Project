import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel

desired_width=1020
pd.set_option('display.width', desired_width)
pd.set_option('display.max.columns',25)

data = pd.read_pickle("/Users/danielsaggau/PycharmProjects/pythonProject/opinions_data.pkl")
data.info()
data.describe()

plain_text = data['plain_text']
author = data['author']
url = data['download_url']
html = data['html']

# converting series to list or str

plain_text = list(plain_text)

# need to fix cleaning function
# cleaning
def clean(plain_text):
    rep={
        '\s +':' '                      #remove whitespace
    }
    return plain_text

clean_text = clean(plain_text)




# encoding
tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")
#inputs = tokenizer_gpt.encode(plain_text, return_tensors = 'tf', truncation = True)

# fix max length error
MAX_LEN = 20

def make_encoding(plain_text, truncation = True, padding = True):  #max_length= max_len
    return tokenizer_gpt(plain_text, truncation = truncation, padding = padding) #max_length = max_len,

encodings = make_encoding(plain_text, tokenizer_gpt)

def make_tfdataset(encodings):
    return tf.data.Dataset.from_tensor_slices(dict(encodings))


inputs = tokenizer_gpt(text, truncation=True)


tfdata = make_tfdataset(encodings)

# split
test = 0.2
batch_size = 2

train = int(len(plain_text) * 1- test)
tf_data = inputs.shuffle(len(plain_text))
data_train = inputs.take(train)
data_test = inputs.skip(train)

# tokenizer


