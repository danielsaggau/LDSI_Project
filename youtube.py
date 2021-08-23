https://www.youtube.com/watch?v=6ORnRAz3gnA

with open("data/training.txt", "r") as fp:
    b = json.load(fp)
b = ''.join(b)

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))


char_indices= dict((c,i) for i, c in enumerate(chars))
indices_char = dict((i,c) for i, c in enumerate(chars))

maxlen = 250
step = 3
sentences = []
next_chars = []

text = b
for i in range(0, len(text)- maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

len(sentences)