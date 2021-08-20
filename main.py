from __future__ import unicode_literals, print_function
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
from keras.preprocessing.text import text_to_word_sequence
from spacy.lang.en import English
from sklearn import model_selection
# set seed

# load data
data = pd.read_pickle("/Users/danielsaggau/PycharmProjects/pythonProject/opinions_data.pkl")
plain_text = data['plain_text']
author = data['author']
url = data['download_url']
html = data['html']
id = data['id']

data_filtered = data[~data['page_count'].isnull()] # remove empty pages
doc_length = data_filtered['plain_text'].str.len()
plain_text = data_filtered['plain_text'] # subset
plain_text = plain_text.str.replace("  ","") # removing whitespaces

text = plain_text
    #text = text.str.replace("\n", " ")
text = text.str.replace("FILED", "")
text = text.str.replace("NOT FOR PUBLICATION", "")
text = text.str.replace("FOR PUBLICATION", "")
text = text.str.replace("FOR PUBLICATION\n", "")
text = text.str.replace("UNITED STATES COURT OF APPEALS", "")
text = text.str.replace("U.S. COURT OF APPEALS", "")
text = text.str.replace("U .S. COURT OF APPEALS", "")
text = text.str.replace("U .S. COURT OF APPEALS", "")
text = text.str.replace("UNITED STATES OF AMERICA", "")
text = text.str.replace("\x0c","")
text = text.str.replace("\uf8fc","")
text = text.str.replace("\uf8fd","")
text = text.str.replace("FOR THE NINTH CIRCUIT", "")
text = text.str.replace("Appeal from the United States District Court", "")

data['date'] = pd.to_datetime(data['date_created'])
data['year'] = data.date.map(lambda x: x.year)
data['year'] = data.year.astype(int)



# convert to sequences
input_sequence = []
output_words = []
input_seq_length = 255

def make_sequence:


# spacy detect sentence boundaries
#reference

nlp = English()
doc = nlp(plain_text)
nlp_plain_text = plain_text.apply(lambda x: nlp(x))
sentences = [sent.string.strip() for sent in nlp_plain_text]

# encoding
tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")

#inputs = tokenizer_gpt.encode(plain_text, return_tensors = 'tf', truncation = True)


tokenizer_gpt.fit_on_texts(plain_text)
#vocabulary:
print(tokenizer.word_index)


plain_text = list(plain_text)
inputs = tokenizer_gpt(text, truncation=True return_tensors ='tf') # fix max length error
inputs = text.apply(tokenizer_gpt)

def make_tfdataset(encodings):
    return tf.data.Dataset.from_tensor_slices(dict(encodings))
data_tf= make_tfdataset(inputs)


# split
test = 0.2
batch_size = 2

train = int(len(plain_text) * 1- test)
tf_data = inputs.shuffle(len(plain_text)) # check for mistake
data_train = tf_data.take(train)
data_test = tf_data.skip(train)

train, test = model_selection.train_test_split(
    plain_text,
    test_size=0.2,
    random_state=42)

# ensure that training and test set dont differ and set unmatched tokens to <unk>