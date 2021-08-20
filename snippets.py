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

# save sequences to file
out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)

# alternative snippet


#https://stackabuse.com/python-for-nlp-deep-learning-text-generation-with-keras/
input_sequence = []
output_words = []
input_seq_length = 255

for i in range(0, n_words - input_seq_length , 1):
    in_seq = macbeth_text_words[i:i + input_seq_length]
    out_seq = macbeth_text_words[i + input_seq_length]
    input_sequence.append([word_2_index[word] for word in in_seq])
    output_words.append(word_2_index[out_seq])

	#
	def top_features_in_doc(Xtr, features, row_id, top_n=15):
		''' Top tfidf features in specific document (matrix row) '''
		xtr_row = Xtr[row_id]
		if type(xtr_row) is not np.ndarray:
			xtr_row = xtr_row.toarray()
		row = np.squeeze(xtr_row)
		return top_tfidf_features(row, features, top_n)


	def span_top_tfidf(spans_txt, spans_tfidf, features, index):
		print('span text:\n' + spans_txt[index] + '\n')
		print(top_features_in_doc(spans_tfidf, features, index))


	vectorizer = TfidfVectorizer(min_df=3)
	vectorizer = vectorizer.fit(plain_text)
	tfidf_features_skl = vectorizer.get_feature_names()

	train_tfidf_skl = vectorizer.transform(plain_text).toarray()
	train_spans_labels = np.array([s['type'] for s in plain_text])

	train_tfidf_skl.shape

	span_top_tfidf(train_tfidf_skl,
				   tfidf_features_skl,
				   random.randint(0, len(train_spans)))
