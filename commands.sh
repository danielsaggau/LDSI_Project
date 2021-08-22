!git clone https://github.com/huggingface/transformers.git
!pip install -q ./transformers

!python /content/transformers/examples/language-modeling/run_language_modeling.py \
--model_type=gpt2 \
--model_name_or_path=distilgpt2 \
--do_train \
--train_data_file=/content/train.txt \
--num_train_epochs 100 \
--output_dir model_output \
--overwrite_output_dir \
--save_steps 20000 \
--per_gpu_train_batch_size 4
