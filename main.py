import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
import datetime

desired_width=1020
pd.set_option('display.width', desired_width)
pd.set_option('display.max.columns',25)

data = pd.read_pickle("/Users/danielsaggau/PycharmProjects/pythonProject/opinions_data.pkl")
data.info()
data.describe()

plain_text = data['plain_text']
author = data['author']
url = data['download_url']
html = data['html']
id = data['id']
# converting series to list
id = list(id)
plain_text = list(plain_text)

documents_by_id = {data['id']: d for d in data['plain_text']}
doc_lengths = [len(data['plain_text']) for id in data()]

data['plain_text'].str.len()

#remove docs without any pages
data_filtered = data_filtered[~data_filtered['page_count'].isnull()]

data = data('\s+', ' ', regex=True)




data_filtered['plain_text'] = dat_filtered['plain_text'].str[60:]

plain_text = data_filtered['plain_text']
#plain_text = plain_text.str.strip()
#plain_text = plain_text.str[700:]
plain_text = plain_text.str.replace("  ","")

plt.hist(doc_lengths, bins=50)
plt.show()


# need to fix cleaning function
# cleaning
def clean(plain_text):
    rep={
        '\s +':' '                      #remove whitespace
    }
    return plain_text

clean_text = clean(plain_text)


cases_df['date_filed'] = pd.to_datetime(cases_df.date_filed)
cases_df['year_filed'] = cases_df.date_filed.map(lambda x: x.year)
cases_df['year_filed'] = cases_df.year_filed.astype(int)

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
