# reference : https://discuss.huggingface.co/t/summarization-on-long-documents/920/26
import re

from ignore_random_notes import tokenizer

RE_SPLITTER = '<eos>'

MODEL_MAX_LEN = 256 # needs to be corrected

def chunk_text(text, num_tok):
    text_sent = \
        [sent.strip() + '.' for sent in re.split(RE_SPLITTER, text) if len(sent) > 1]

    # calculate number of tokens per sentence
    num_tok_sent = [len(tokenizer.tokenize(sent)) for sent in text_sent]

    # calculate chunk dimension to fit into model
    n = int(np.ceil(num_tok / MODEL_MAX_LEN))
    len_chunk = int(num_tok / n)

    # get a more uniform splitting to avoid splits
    # which are too short at the end
    if len_chunk + 50 > MODEL_MAX_LEN:
        len_chunk = int(num_tok / (n + 1))

    len_curr = 0
    text_curr = []
    text_chunk = []
    for te, len_sent in zip(text_sent, num_tok_sent):

        if len_curr + len_sent < len_chunk:
            text_curr.append(te)
            len_curr += len_sent

        elif len_curr + len_sent >= MODEL_MAX_LEN:
            text_chunk.append(text_curr)

            text_curr = [te]
            len_curr = len_sent

        else:  # >= len_chunk && < MODEL_MAX_LEN
            text_curr.append(te)
            text_chunk.append(text_curr)

            text_curr = []
            len_curr = 0

    if len_curr > 0:
        text_chunk.append(text_curr)

    return text_chunk

chunky = chunk_text(text, 256)

import numpy as np
np.savetxt("file_name.csv", chunky, delimiter="]", fmt='%s')