from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
from transformers import AutoConfig, AutoModelForCausalLM
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from transformers import AutoConfig, AutoModelForCausalLM,AutoTokenizer
import json
from datasets import load_dataset
import csv
import torch



model_checkpoint = "gpt2"
tokenizer_checkpoint = "sgugger/gpt2-like-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

#datasets = load_dataset("text", data_files={"train": 'data/training.txt'})
tokenizer.pad_token = tokenizer.eos_token


input_ids = [tokenizer.encode(sent, return_tensors="pt",truncation=True,padding='max_length') for sent in flat_list]

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01
)

config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_config(config)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tfdata,
)

trainer.train()

trainer.train()
