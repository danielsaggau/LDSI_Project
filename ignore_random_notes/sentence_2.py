def split_on_breaks(doc):
    start = 0
    seen_break = False
    for word in doc:
        if seen_break:
            yield doc[start:word.i-1]
            start = word.i
            seen_break = False
        elif word.text == '<eos>':
            seen_break = True
    if start < len(doc):
        yield doc[start:len(doc)]

nlp.add_pipe('split_on_breaks', before='parser')

def get_sentences(text):
    doc = nlp(text)
    return (list(doc.sents))