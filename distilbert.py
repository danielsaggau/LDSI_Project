# https://colab.research.google.com/github/peterbayerle/huggingface_notebook/blob/main/distilbert_tf.ipynb#scrollTo=rVU_LoASQMcp
N_EPOCHS = 2

model_bert = TFGPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=tkzr.eos_token_id)
optimizer = optimizers.Adam(learning_rate=3e-5)
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model_bert.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model_bert.fit(tfdataset_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS)

input_ids = tkzr.encode(x, return_tensors="tf")

greedy_output = model_bert.generate(input_ids, max_length=100)
print("Output:\n" + 100 * '-')
print(tkzr.decode(greedy_output[0], skip_special_tokens=True))

sample_outputs = model_bert.generate(
    input_ids,
    max_length=25,
    num_beams=5,
    early_stopping = True,
    num_return_sequences=3,
)
print("Output:\n" + 100 * '-')
print(tkzr.decode(sample_outputs[0], skip_special_tokens=True))