

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# inputs= tokenizer.encode_plus(flat_list, padding='max_length', truncation=True,return_tensors="tf")

input_ids = [tokenizer.encode(sent, return_tensors="tf",truncation=True,padding='max_length') for sent in flat_list]
tokenized_text = input_ids

examples = []
block_size = 100
for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
    examples.append(tokenized_text[i:i+block_size])

inputs, labels = [], []
for ex in examples:
    inputs.append(ex[:-1])
    labels.append(ex[1:])

from transformers import GPT2Config,TFGPT2LMHeadModel
config = GPT2Config.from_pretrained('distilgpt2')

model = TFGPT2LMHeadModel.from_pretrained('distilgpt2', pad_token_id=tokenizer.eos_token)
# splitting into test and training set

dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

# reference: https://colab.research.google.com/github/peterbayerle/huggingface_notebook/blob/main/distilbert_tf.ipynb#scrollTo=fKTJqUF5R-o4

TEST_SPLIT = 0.2
BATCH_SIZE = 2

train_size = int(len(inputs) * (1-TEST_SPLIT))

dataset = dataset.shuffle(len(inputs))
dataset_train = dataset.take(train_size)
dataset_test = dataset.skip(train_size)

dataset_train = dataset_train.batch(BATCH_SIZE)
dataset_test = dataset_test.batch(BATCH_SIZE)


BATCH_SIZE = 2
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
model.fit(dataset, epochs=20)


