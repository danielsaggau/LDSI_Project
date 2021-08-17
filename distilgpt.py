# encoding
tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")
#inputs = tokenizer_gpt.encode(plain_text, return_tensors = 'tf', truncation = True)
inputs = tokenizer_gpt(text, truncation=True)
# fix max length error

def make_tfdataset(encodings):
    return tf.data.Dataset.from_tensor_slices(dict(encodings))

# split
test = 0.2
batch_size = 2

train = int(len(plain_text) * 1- test)
tf_data = inputs.shuffle(len(plain_text)) # check for mistake
data_train = tf_data.take(train)
data_test = tf_data.skip(train)

# Instantiate Model

model_gpt = TFGPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)

model_gpt.train(text,
         line_by_line=False,
         num_steps=36000,
         generate_every=1000,
         save_every=1000,
         learning_rate=1e-4,
         batch_size=2,
         )

# Generate Text

sample_outputs = model_gpt.generate(
    input_ids_gpt,
    do_sample=True,
    max_length=256,
    batch_size=2,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
    repetition_penalty = 1.5
)
print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

# Flesch–Kincaid readability score


# Performance Evaluation

# ROUGE

# BLEU


# BLEURT
