import pytest
from latent_cartography.token_utils import get_latin_word_tokens

from transformers import AutoTokenizer


# Define the model names for LLaMA-2, Mistral, and GPT-2
model_names = {
    "llama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "gpt2": "gpt2",
    # "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
}


# Create parameterization for the tokenizers
@pytest.fixture(params=model_names.values(), ids=model_names.keys())
def tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


# Create parameterization for the tokens
@pytest.fixture(params=model_names.values(), ids=model_names.keys())
def tokens(request):
    tokenizer = AutoTokenizer.from_pretrained(request.param)
    return get_latin_word_tokens(tokenizer)


def test_get_word_tokens():
    # Test the get_word_tokens function with each tokenizer
    for model_name in model_names.values():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        word_tokens = get_latin_word_tokens(tokenizer)
        assert isinstance(word_tokens, list), f"Word tokens for {model_name} should be a list"
        assert len(word_tokens) > 0, f"Failed to get word tokens for {model_name}"
        assert len(word_tokens) >= 20000, f"Expected at least 20,000 word tokens for {model_name}, got {len(word_tokens)}"


def test_word_spaces(tokenizer, tokens):
    # Chech that, given a string of tokens to decode, the tokenizer inserts spaces between them
    tokens = tokens[:10]
    decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
    assert " " in decoded


def test_cloze_insertion(tokenizer, tokens):
    # Tests some tokens when inserted into sentences, that spacing is corrent
    cloze = "The MASK word."
    cloze_tokens = tokenizer.encode(cloze)
    mask_token = tokenizer.encode("MASK")[0]
    assert mask_token in cloze_tokens

    mask_id = cloze_tokens.index(mask_token)

    # Replace the MASK token with a word token
    for token in tokens[:10]:
        cloze_tokens[mask_id] = token
        decoded = tokenizer.decode(cloze_tokens)
        assert decoded.startswith("The ") and decoded.endswith(" word.")
        assert decoded.count(" ") == 2
