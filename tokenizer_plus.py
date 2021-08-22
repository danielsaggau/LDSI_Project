import pandas as pd
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel

with open("/Users/danielsaggau/PycharmProjects/pythonProject/data/output.txt", "r", encoding ="utf-8") as f:
    text = f.read().split("\n")

map_object = map(str.replace("\*"," "), text)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import spacy
from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")

doc = nlp(text[:1000000])
sentences = list(sents.text for sents in doc.sents) # ensure that we get strings and not spans
sentences[812]

# adding end of sentence token for tokenizer

fr_text_new = []
for sent in sentences:
    sent_new = " ".join([sent, '<eos>'])
    fr_text_new.append(sent_new)
    print("After adding tokens: ", sent_new, '\n')

fr_text_new[20:23]
tokenizer.add_special_tokens({'pad_token': '<eos>'})
pad_token='<eos>'

with open("data/file.txt", "w") as output:
    output.write(str(fr_text_new))

with open("/Users/danielsaggau/PycharmProjects/pythonProject/data/file.txt", "r", encoding ="utf-8") as f:
    text = f.read()

# splitting data


def encode(text):
    return(tokenizer(text['text'], truncation =True, padding ='max_length'))

fr_text_new.map(encode, batched = True)


max_length= 256
data= pd.DataFrame(fr_text_new)
inputs = tokenizer(text, max_length= 256, padding = True, truncation=True, return_tensors ="tf")

print(inputs)

tokenizer.decode(inputs['input_ids'])