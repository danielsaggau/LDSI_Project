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

inputs = tokenizer(sentences, max_length= 256, padding = True, truncation=True, return_tensors ="tf",pad_token=eos_token)

print(inputs)

def tokenizer(sentences):
    return()

tokenizer.decode(inputs)