def join_tokens(text_in):
    text_out = u" " + ' '.join(text_in)
    while text_out.count(u"  ") > 0:
        text_out = text_out.replace(u"  ", u" ")
    return text_out

for sent in doc.sents:
        sentence_text = join_tokens(sentence_text)
        if sentence_len < MIN_SENTENCE_LEN or '\n' in sentence_text:
            sentence_group = []
            sentence_group.append(sentence_text)

            s_beginning = join_tokens(sentence_group[:BEGINNING_LEN]
file = []
for sent in doc.sents:
    sentence_len =
if sentence_len > MIN_SENTENCE_LEN or '\n' in sentence_text:
file.append()

for doc in doc_sents:
    sentence_text = []
    for token in doc:
        sentence_text.append(token.text)
    sentence_len = len(sentence_text)
    sentence_text = join_tokens(sentence_text)