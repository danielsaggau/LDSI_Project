import pandas as pd
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
import re
import tokenizer
import numpy as np
import json
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


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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



with open("data/training.json", "w") as fp:
     json.dump(batch, fp)


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

dat = np.loadtxt("data/sentence_batches.csv")

chunky = chunk_text(batch, 256)

tokenizer(batch_sentences, padding='max_length', truncation=True)