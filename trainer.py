from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel, GPT2Tokenizer, TFDistilBertPreTrainedModel, TFGPT2LMHeadModel
from transformers import AutoConfig, AutoModelForCausalLM
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from transformers import AutoConfig, AutoModelForCausalLM,AutoTokenizer
import json
from datasets import load_dataset

datasets = load_dataset("text", data_files={"train": 'data/training.txt'})


datasets["train"][0]

model_checkpoint = "gpt2"
tokenizer_checkpoint = "sgugger/gpt2-like-tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    push_to_hub_model_id=f"{model_checkpoint}-wikitext2",
)

config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_config(config)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

trainer.train()

trainer.train()
