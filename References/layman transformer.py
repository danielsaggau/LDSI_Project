from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)

examp = tokenizer.encode('The district court properly granted summary judgment on plaintiffs’ claim alleging municipal', return_tensors ="tf")
print(examp)
tokenizer.decode(examp["input_ids"])

output = model(examp)
logits = output.logits

tokenizer.decode(output)

greedy_output = model.generate(examp, max_length=50)
greedy_output = model.generate(input_ids, max_length=50)



