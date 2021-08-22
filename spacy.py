import spacy
import io

spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')

from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")

# adding special tokens 

#nlp.tokenizer.add_special_case('Vet. App.', [{ORTH: 'Vet. App.'}])
#nlp.tokenizer.add_special_case('Fed. Cir.', [{ORTH: 'Fed. Cir.'}])

raw_text = io.open("/Users/danielsaggau/PycharmProjects/pythonProject/data/output.txt", "r", encoding='utf8').read()

#doc = list(nlp.pipe(raw_text, disable=["tagger","ner", "attribute_ruler", "lemmatizer"]))
doc = list(nlp.pipe(raw_text))
doc = nlp(raw_text[:1000000])
sents = list(doc.sents)
sentence_text = []
sentence_len = []

for sent in doc.sents:
    sentence_text.append(token.text)
    sentence_len = len(sentence_text)

for token in sents:
    sentence_text = join_tokens(sentence_text)

for sent in doc.sents:
    for token in sent:
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

fr_text_new = []
for sent in sents:
    sent_new = " ".join([sent, 'eos'])
    # Append the modified sentence to fr_text_new
    fr_text_new.append(sent_new)
    # Print sentence after adding tokens
    print("After adding tokens: ", sent_new, '\n')