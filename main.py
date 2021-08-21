from __future__ import unicode_literals, print_function
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
#from keras.preprocessing.text import text_to_word_sequence
from spacy.lang.en import English
#from sklearn import model_selection
# set seed

# load data
data = pd.read_pickle("/data/opinions_data.pkl")


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

# save as txt
text.to_csv('/Users/danielsaggau/PycharmProjects/pythonProject/output.txt', sep='\n', index=False)
# load text
raw_text = io.open("/data/output.txt", "r", encoding='utf8').read()
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')
doc = nlp(raw_text[:1000000])
# max value

doc_sents = [sent for sent in doc.sents]
token_sents = [token for sent in doc.sents] # bound to try out
sentences = pd.DataFrame(data={"col1": doc_sents})
sentences.to_csv('/Users/danielsaggau/PycharmProjects/pythonProject/sentence.csv', sep ="\n")
print(doc_sents[100])

print(sents_list)
print([token.text for token in doc])

def make_sequence:
# spacy detect sentence boundaries
#reference

nlp = English()
doc = nlp(plain_text)

for doc in doc.sents:
    sentence_text = []
    for token in doc:
        sentence_text.append(token.text)
    sentence_len = len(sentence_text)
    sentence_text = join_tokens(sentence_text)

# improving segmentation by adding exceptions and special cases

length = 255 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))





# encoding
tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")
#inputs = tokenizer_gpt.encode(plain_text, return_tensors = 'tf', truncation = True)

plain_text = list(plain_text)

# ensure that training and test set dont differ and set unmatched tokens to <unk>