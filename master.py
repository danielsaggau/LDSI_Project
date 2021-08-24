import pandas as pd
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
import re
import tokenizer
import numpy as np
import json
import tensorflow as tf
# removing xml markup, symbols and starting text
text = open("/Users/danielsaggau/PycharmProjects/pythonProject/data/output.txt", "r").read()
text = text.replace("plain_text", " ")
text = text.replace("FILED", "")
text = text.replace("NOT FOR PUBLICATION", "")
text = text.replace("\n"," ")
text = text.replace("**", " ")
text = text.replace("*", " ")
text = text.replace("  ", " ")
text = text.replace("\uf8fe", " ")

from spacy.lang.en import English
from spacy.attrs import ORTH, NORM
nlp = English()
nlp.add_pipe("sentencizer")

# adding special tokens
nlp.tokenizer.add_special_case('Fed. Cir.', [{ORTH: 'Fed. Cir.'}])
nlp.tokenizer.add_special_case('9th Cir.', [{ORTH: '9th Cir.'}])
nlp.tokenizer.add_special_case('See Fed.', [{ORTH: 'See Fed.'}])
nlp.tokenizer.add_special_case('IN NO.', [{ORTH: 'IN NO.'}])
nlp.tokenizer.add_special_case('R.App.', [{ORTH: 'R.App.'}])
nlp.tokenizer.add_special_case('R. App.', [{ORTH: 'R. App.'}])
nlp.tokenizer.add_special_case('R. Civ.', [{ORTH: 'R. Civ.'}])
nlp.tokenizer.add_special_case('Dkt., No.', [{ORTH: 'Dkt., No.'}])
nlp.tokenizer.add_special_case('Dkt.', [{ORTH: 'Dkt.'}])
nlp.tokenizer.add_special_case('Dkt.,No.', [{ORTH: 'Dkt.,No.'}])
nlp.tokenizer.add_special_case('et al.', [{ORTH: 'et al.'}])
nlp.tokenizer.add_special_case('D.C. No.', [{ORTH: 'D.C. No.'}])
nlp.tokenizer.add_special_case('No. ,', [{ORTH: 'No. ,'}])
nlp.tokenizer.add_special_case('No.', [{ORTH: 'No.'}])
nlp.tokenizer.add_special_case('Nos.', [{ORTH: 'Nos.'}])
nlp.tokenizer.add_special_case('Fed.', [{ORTH: 'Fed.'}])

doc = nlp(text[:1000000])
sentences = list(sents.text for sents in doc.sents) # ensure that we get strings and not spans

batch = []
for sent in sentences:
    sent_new = " ".join([sent, '<eos>'])
    batch.append(sent_new)
    print("After adding tokens: ", sent_new, '\n')



with open("data/training.txt", "w") as fp:
     json.dump(batch, fp)

with open("data/training.txt", "r") as fp:
     b = json.load(fp)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
RE_SPLITTER = '<eos>'

MODEL_MAX_LEN = 256 # needs to be corrected

sent= []

def chunk_text(text, num_tok):
    text_sent = \
        [sent.strip() + '.' for sent in re.split(RE_SPLITTER, text) if len(sent) > 1]

    # calculate number of tokens per sentence
    num_tok_sent = [len(tokenizer.tokenize(sent)) for sent in text_sent]

    # calculate chunk dimension to fit into model
    n = int(np.ceil(num_tok / MODEL_MAX_LEN))
    len_chunk = int(num_tok / n)

    # get a more uniform splitting to avoid splits
    # which are too short at the end
    if len_chunk + 50 > MODEL_MAX_LEN:
        len_chunk = int(num_tok / (n + 1))

    len_curr = 0
    text_curr = []
    text_chunk = []
    for te, len_sent in zip(text_sent, num_tok_sent):

        if len_curr + len_sent < len_chunk:
            text_curr.append(te)
            len_curr += len_sent

        elif len_curr + len_sent >= MODEL_MAX_LEN:
            text_chunk.append(text_curr)

            text_curr = [te]
            len_curr = len_sent

        else:  # >= len_chunk && < MODEL_MAX_LEN
            text_curr.append(te)
            text_chunk.append(text_curr)

            text_curr = []
            len_curr = 0

    if len_curr > 0:
        text_chunk.append(text_curr)

    return text_chunk

b = ''.join(b)
chunky = chunk_text(b, 256)

with open("data/chunks.txt", "r") as fp:
    b = json.load(fp)


with open("data/sentence_chunks.json", "w") as fp:
    json.dump(chunky, fp)

flat_list = [item for sublist in chunky for item in sublist]

with open("data/flat_list.json", "w") as fp:
    json.dump(flat_list, fp)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# inputs= tokenizer.encode_plus(flat_list, padding='max_length', truncation=True,return_tensors="tf")

input_ids = [tokenizer.encode(sent, return_tensors="tf",truncation=True,padding='max_length') for sent in flat_list]
tokenized_text = input_ids

examples = []
block_size = 100
for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
    examples.append(tokenized_text[i:i+block_size])

inputs, labels = [], []
for ex in examples:
    inputs.append(ex[:-1])
    labels.append(ex[1:])

from transformers import GPT2Config,TFGPT2LMHeadModel
config = GPT2Config.from_pretrained('distilgpt2')

model = TFGPT2LMHeadModel.from_pretrained('distilgpt2', pad_token_id=tokenizer.eos_token)
# splitting into test and training set

dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

# reference: https://colab.research.google.com/github/peterbayerle/huggingface_notebook/blob/main/distilbert_tf.ipynb#scrollTo=fKTJqUF5R-o4

TEST_SPLIT = 0.2
BATCH_SIZE = 2

train_size = int(len(inputs) * (1-TEST_SPLIT))

dataset = dataset.shuffle(len(inputs))
dataset_train = dataset.take(train_size)
dataset_test = dataset.skip(train_size)

dataset_train = dataset_train.batch(BATCH_SIZE)
dataset_test = dataset_test.batch(BATCH_SIZE)


BATCH_SIZE = 2
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
model.fit(dataset, epochs=20)


