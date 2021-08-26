# removing xml markup, symbols and starting text
text = open("/Users/danielsaggau/PycharmProjects/pythonProject/data/output.txt", "r").read()

nlp = English()
nlp.add_pipe("sentencizer")

# adding special tokens
nlp.tokenizer.add_special_case('9th Cir.', [{ORTH: '9th Cir.'}])
nlp.tokenizer.add_special_case('Cir.', [{ORTH: 'Cir.'}])
nlp.tokenizer.add_special_case('Fed.', [{ORTH: 'Fed.'}])
nlp.tokenizer.add_special_case('NO.', [{ORTH: 'NO.'}])
nlp.tokenizer.add_special_case('App.', [{ORTH: 'App.'}])
nlp.tokenizer.add_special_case('Civ.', [{ORTH: 'Civ.'}])
nlp.tokenizer.add_special_case('No.', [{ORTH: 'No.'}])
nlp.tokenizer.add_special_case('Dkt.', [{ORTH: 'Dkt.'}])
nlp.tokenizer.add_special_case('et al.', [{ORTH: 'et al.'}])
nlp.tokenizer.add_special_case('D.C.', [{ORTH: 'D.C.'}])
nlp.tokenizer.add_special_case('No.', [{ORTH: 'No.'}])
nlp.tokenizer.add_special_case('Nos.', [{ORTH: 'Nos.'}])
nlp.tokenizer.add_special_case('Fed.', [{ORTH: 'Fed.'}])
nlp.tokenizer.add_special_case('U.S.C.', [{ORTH: 'U.S.C.'}])
nlp.tokenizer.add_special_case(' F.', [{ORTH: 'F.'}])
nlp.tokenizer.add_special_case(' R.', [{ORTH: 'R.'}])
nlp.tokenizer.add_special_case(' P.', [{ORTH: 'P.'}])
nlp.tokenizer.add_special_case(' n.', [{ORTH: ' n.'}])
nlp.tokenizer.add_special_case(' v.', [{ORTH: ' v.'}]) # versus
nlp.tokenizer.add_special_case('Univ.', [{ORTH: 'Univ.'}])
nlp.tokenizer.add_special_case('Jr.', [{ORTH: 'Jr.'}])
nlp.tokenizer.add_special_case('I.N.S.', [{ORTH: 'I.N.S.'}])
nlp.tokenizer.add_special_case('OR.', [{ORTH: 'OR.'}])
nlp.tokenizer.add_special_case('REV.', [{ORTH: 'REV.'}])
nlp.tokenizer.add_special_case('STAT.', [{ORTH: 'STAT.'}])
doc = nlp(text[:1000000])
sentences = list(sents.text for sents in doc.sents) # ensure that we get strings and not spans

batch = []
for sent in sentences:
    sent_new = " ".join([sent, '<eos>'])
    batch.append(sent_new)
    print("After adding tokens: ", sent_new, '\n')



with open("data/training.txt", "w") as fp:
     json.dump(batch, fp)

with open("data/training.txt", "r") as fp:
     b = json.load(fp)

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

b = ''.join(b)
chunky = chunk_text(b, 256)

with open("data/chunks.txt", "r") as fp:
    b = json.load(fp)


with open("data/sentence_chunks.json", "w") as fp:
    json.dump(chunky, fp)

flat_list = [item for sublist in chunky for item in sublist]

with open("data/flat_list.json", "w") as fp:
    json.dump(flat_list, fp)
