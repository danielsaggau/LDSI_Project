MAX_LEN = 20

def construct_encodings(x, tokenizer, max_len, trucation=True, padding=True):
    return tokenizer(x, max_length=max_len, truncation=trucation, padding=padding)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
encodings = construct_encodings(data, tokenizer, max_len=MAX_LEN)

def construct_tfdataset(encodings, y=None):
    if y:
        return tf.data.Dataset.from_tensor_slices((dict(encodings), y))
    else:
        # this case is used when making predictions on unseen samples after training
        return tf.data.Dataset.from_tensor_slices(dict(encodings))

tfdataset = construct_tfdataset(encodings)


TEST_SPLIT = 0.2
BATCH_SIZE = 2

train_size = int(len(data) * (1-TEST_SPLIT))

tfdataset = tfdataset.shuffle(len(data))
tfdataset_train = tfdataset.take(train_size)
#tfdataset_test = tfdataset.skip(train_size)

tfdataset_train = tfdataset_train.batch(BATCH_SIZE)
#tfdataset_test = tfdataset_test.batch(BATCH_SIZE)