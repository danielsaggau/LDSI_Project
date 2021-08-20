import io
print("Initializing spaCy")
nlp = English()
NUM_EXPERIMENTS = 100
MIN_SENTENCE_LEN = 10
NUM_SENTENCES = 4
BEGINNING_LEN = 1



print("Reading document")
raw_text = open("/Users/danielsaggau/PycharmProjects/pythonProject/output.txt", "r", encoding='utf8').read()
raw_text = open("/Users/danielsaggau/PycharmProjects/pythonProject/output.txt", "r").read()
nlp.add_pipe('sentencizer')

doc = nlp(raw_text[:1000000])

def join_tokens(text_in):
    text_out = u" " + ' '.join(text_in)
    while text_out.count(u"  ") > 0:
        text_out = text_out.replace(u"  ", u" ")
    return text_out

print("Start")
sentences = []
sentence_group = []
sentences_beginning = []
sentences_end = []

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
    output_df.to_csv('/Users/danielsaggau/PycharmProjects/pythonProject/okey.csv', index=False, header=True)