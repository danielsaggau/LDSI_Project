for token in doc.sents:
    sentence_text = []
    sentence_text.append(token.text)
sentence_len = len(sentence_text)
sentence_text = join_tokens(sentence_text)

for sent in doc.sents:
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