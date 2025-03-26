import json
import os
import re
from typing import Callable

import torch
from fire import Fire
from transformers import AutoModel, AutoTokenizer

# from tokenizer_changer import TokenizerChanger
from .tokenizer_changer import TokenizerChanger

# Below is the list of functions to filter out the unused tokens.
# The tokens that return True will be removed from the tokenizer.

japanese_pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
vietnamese_pattern = re.compile(r"[àáâãèéêìíòóôõùúýđÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĐ\u0300-\u036f]")
russian_pattern = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")
persian_pattern = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
german_specific_chars = set(["ä", "ö", "ü", "ß", "Ä", "Ö", "Ü", "ẞ"])
french_specific_chars = set(["é", "è", "ê", "à", "ç", "ô", "ù", "ï", "â", "œ", "æ"])
indonesian_special_chars = set(["é", "è", "ê", "à", "ç", "ô", "ù", "ï", "â", "ñ"])
romanian_special_chars = set(["ă", "â", "î", "ș", "ț", "Ă", "Â", "Î", "Ș", "Ț"])


def contains_japanese(token: str) -> bool:
    return bool(japanese_pattern.search(token))


def contains_chinese(token: str) -> bool:
    return bool(chinese_pattern.search(token))


def contains_vietnamese(token: str) -> bool:
    return bool(vietnamese_pattern.search(token))


def contains_russian(token: str) -> bool:
    return bool(russian_pattern.search(token))


def contains_persian(token: str) -> bool:
    return bool(persian_pattern.search(token))


def contains_german(token: str) -> bool:
    return any(char in german_specific_chars for char in token)


def contains_french(token: str) -> bool:
    return any(char in french_specific_chars for char in token)


def contains_indonesian(token: str) -> bool:
    return any(char in indonesian_special_chars for char in token)


def contains_romanian(token: str) -> bool:
    return any(char in romanian_special_chars for char in token)


# Append filtering functions here.
# NOTE: If a filter returns True for a token, the token will be removed.
filterings_for_multilingual = [
    contains_japanese,
    contains_chinese,
    contains_vietnamese,
    contains_russian,
    contains_persian,
    contains_german,
    contains_french,
    contains_indonesian,
    contains_romanian,
]


def trim_sentence_piece_tokenizer(
    origin_model_path: str,
    trimmed_model_path: str,
    filterings_for_removal: list[Callable] = filterings_for_multilingual,
) -> None:
    """
    Trimming for the SentencePiece tokenizer.
    Redundant tokens will be removed using the filtering functions.

    Args:
        origin_model_path (str): the dirpath of the tokenizer as a base.
        trimmed_model_path (str): the dirpath of the trimmed tokenizer.
    """
    os.makedirs(trimmed_model_path, exist_ok=True)

    model_o = AutoModel.from_pretrained(origin_model_path)
    tokenizer_o = AutoTokenizer.from_pretrained(origin_model_path)
    model_state_o = json.loads(tokenizer_o.backend_tokenizer.model.__getstate__())
    vocab = list(
        filter(
            lambda token: any(filtering(token) for filtering in filterings_for_removal),
            [token for token, _ in model_state_o["vocab"]],
        ),
    )
    trimmer = TokenizerChanger(tokenizer=tokenizer_o, space_sign="▁")
    trimmer.delete_tokens(vocab, delete_merges=False)
    trimmer.save_tokenizer(trimmed_model_path)

    tokenizer_t = AutoTokenizer.from_pretrained(trimmed_model_path)
    model_state_t = json.loads(tokenizer_t.backend_tokenizer.model.__getstate__())

    token2id_o = dict(
        (token, id_) for id_, (token, _) in enumerate(model_state_o["vocab"])
    )
    token2id_t = dict(
        (token, id_) for id_, (token, _) in enumerate(model_state_t["vocab"])
    )

    embeddings_o = model_o.get_input_embeddings().weight.data
    embeddings_t = embeddings_o[[token2id_o[token] for token in token2id_t.keys()]]
    new_embedding_layer = torch.nn.Embedding.from_pretrained(embeddings_t)

    model_o.set_input_embeddings(new_embedding_layer)
    model_o.resize_token_embeddings(len(token2id_t))
    model_o.save_pretrained(trimmed_model_path)


if __name__ == "__main__":
    Fire(trim_sentence_piece_tokenizer)
