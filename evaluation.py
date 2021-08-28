import spacy
from spacy_readability import Readability

nlp = spacy.load('en')
read = Readability(nlp)
nlp.add_pipe(read, last=True)
doc = nlp(generation)
doc._.flesch_kincaid_grade_level
doc._.flesch_kincaid_reading_ease
doc._.dale_chall