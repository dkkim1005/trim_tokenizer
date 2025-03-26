# Trim tokenizer
Reducing embeddings by trimming unused tokens to reduce the embedding size.


## build
``` sh
$ poetry install
$ poetry shell  # switch to the virtual env
```

## how to use?
``` python3
from trim_tokenizer import trim_sentence_piece_tokenizer

if __name__ == "__main__":
    # Suppose that we want to remove some multilingual tokens:
    trim_sentence_piece_tokenizer(
        origin_model_path="./e5-small-v2",
        trimmed_model_path="./e5-small-v2-trimmed",
    )
```
