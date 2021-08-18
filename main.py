import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
from keras.preprocessing.text import text_to_word_sequence

# set seed

# load data
data = pd.read_pickle("/Users/danielsaggau/PycharmProjects/pythonProject/opinions_data.pkl")
plain_text = data['plain_text']
author = data['author']
url = data['download_url']
html = data['html']
id = data['id']

data_filtered = data[~data['page_count'].isnull()] # remove empty pages
doc_length = data_filtered['plain_text'].str.len()
plain_text = data_filtered['plain_text'] # subset
plain_text = plain_text.str.replace("  ","") # removing whitespaces

text = plain_text.str.replace("FILED", "")
text = text.str.replace("NOT FOR PUBLICATION", "")
text = text.str.replace("UNITED STATES COURT OF APPEALS", "")
text = text.str.replace("U.S. COURT OF APPEALS", "")
text = text.str.replace("U.S. COURT OF APPEALS", "")


data['date_filed'] = pd.to_datetime(data['date_created'])
data['year_filed'] = data.date_filed.map(lambda x: x.year)
data['year_filed'] = data.year_filed.astype(int)

# convert to sequences
# doubtful this is smart/necessary
text = plain_text.apply(text_to_word_sequence)


# encoding
tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")
#inputs = tokenizer_gpt.encode(plain_text, return_tensors = 'tf', truncation = True)
plain_text = list(plain_text)
inputs = tokenizer_gpt(plain_text, truncation=True) # fix max length error

plain_text = list(plain_text)


# save sequences to file
out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)

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