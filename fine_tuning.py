from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
inputs = tokenizer.encode(raw_text, return_tensors = 'tf', truncation = True)
inputs = tokenizer(text, truncation=True)
gpt2 = TFGPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)
sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=f/Users/danielsaggau/PycharmProjects/pythonProject/data/output.txt,
              model_name='distilgpt2',
              steps=1000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=200,
              save_every=500
              )