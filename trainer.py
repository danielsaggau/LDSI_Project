from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, AutoModelForCausalLM
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from transformers import AutoConfig, AutoModelForCausalLM

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

trainer.train()
