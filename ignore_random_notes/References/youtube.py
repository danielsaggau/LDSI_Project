from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
https://www.youtube.com/watch?v=6ORnRAz3gnA

with open("../../data/training.txt", "r") as fp:
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

############################
import tensorflow as tf
import json

with open("../../data/flat_list.json", 'r') as f:
    datastore = json.load(f)

trainingsize = 4000
training_data = datastore[: trainingsize]
print(len(training_data))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
train_embeddings = tokenizer(training_data, truncation = True, padding = True)
print(len(train_embeddings))

tfdata = tf.data.Dataset.from_tensor_slices(dict(train_embeddings))
model = TFGPT2LMHeadModel.from_pretrained('distilgpt2', pad_token_id=tokenizer.eos_token)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer = optimizer, loss = model.compute_loss, metrics['accuracy'])