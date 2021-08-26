from pathlib import Path

def read_corpus(split_dir):
    split_dir = Path(split_dir)
    texts = []
    for text_file in (split_dir).iterdir():
            texts.append(text_file.read_text())
    return texts

train_texts= read_corpus('../data/flat_list.json')