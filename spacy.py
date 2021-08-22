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
doc = nlp(raw_text)