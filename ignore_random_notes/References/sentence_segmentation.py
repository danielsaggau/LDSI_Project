# This code heavily copies from the sentence segmentation from lazar et al (2019) for comparability
# load packages
import io
import pandas as pd
import spacy
import random
random.seed(10)
# download english directory for pycharm
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')
# select number of sentences : identical to selection by Lazar
NUM_EXPERIMENTS = 500
MIN_SENTENCE_LEN = 10
NUM_SENTENCES = 4
BEGINNING_LEN = 1

raw_text = open("/data/output.txt", "r").read()
#doc = nlp(raw_text[1000000:2000000])
doc = list(nlp(raw_text[:1000000]))

def join_tokens(text_in):
    text_out = u" " + ' '.join(text_in)
    while text_out.count(u"  ") > 0:
        text_out = text_out.replace(u"  ", u" ")
    return text_out

sentences = []
sentence_group = []
sentences_beginning = []
sentences_end = []
sentence_len = []

while True:
    for sent in doc.sents:
        sentence_text = []
        for token in sent:
            sentence_text.append(token.text)
        sentence_len = len(sentence_text)
        sentence_text = join_tokens(sentence_text)
        if sentence_len < MIN_SENTENCE_LEN or '\n' in sentence_text:
            sentence_group = []
            continue
        sentence_group.append(sentence_text)
        if len(sentence_group) >= NUM_SENTENCES:
            sentences.append(sentence_group)
            s_beginning = join_tokens(sentence_group[:BEGINNING_LEN])
            s_end = join_tokens(sentence_group[BEGINNING_LEN:])
            sentences_beginning.append(s_beginning)
            sentences_end.append(s_end)
            sentence_group = []

        if len(sentences) >= NUM_EXPERIMENTS:
            break
    break



export_to_csv = True
if export_to_csv:
    output_dict = {"beginning": sentences_beginning, "true_end": sentences_end}
    output_df = pd.DataFrame(output_dict, columns=["beginning", "true_end"])
    output_df.to_csv('/Users/danielsaggau/PycharmProjects/pythonProject/tried.csv', index=False, header=True)