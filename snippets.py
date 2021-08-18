#import datetime

lines = plain_text.str.split('\n')
# organize into sequences of tokens
length = 255 + 1
lines = list()
for i in range(length, len(inputs)):
	seq = tokens[i-length:i]
	line = ' '.join(seq)
	lines.append(line)
print('Total Sequences: %d' % len(sequences))

# organize into sequences of tokens
length = 255 + 1
seq = list()
for i in range(length, len(text)):
	seq = text[i-length:i]
	line = ' '.join(seq)
	seq.append(line)
print('Total Sequences: %d' % len(seq))

# alternative snippet
#https://stackabuse.com/python-for-nlp-deep-learning-text-generation-with-keras/
input_sequence = []
output_words = []
input_seq_length = 100

for i in range(0, n_words - input_seq_length , 1):
    in_seq = macbeth_text_words[i:i + input_seq_length]
    out_seq = macbeth_text_words[i + input_seq_length]
    input_sequence.append([word_2_index[word] for word in in_seq])
    output_words.append(word_2_index[out_seq])