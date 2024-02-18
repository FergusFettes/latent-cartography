from string import ascii_letters
import re
from transformers import PreTrainedTokenizer


def get_latin_word_tokens(tokenizer: PreTrainedTokenizer):
    """
    Retrieves all tokens that represent words from the given tokenizer.
    Tokens that are words may have a space before them if that's how the tokenizer operates.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer from which to retrieve word tokens.

    Returns:
        list: A list of word tokens.
    """
    all_tokens = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    letters = ascii_letters + " '"

    latin_word_tokens = [normalize_token_spacing(tokenizer, token) for token in all_tokens if re.match(f"^[{letters}]+$", token)]
    latin_word_tokens = [token for token in latin_word_tokens if token.strip()]
    return latin_word_tokens


def normalize_token_spacing(tokenizer: PreTrainedTokenizer, token: str) -> str:
    """
    Normalizes the spacing for a token based on the tokenizer's behavior.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to check for spacing behavior.
        token (str): The token to normalize.

    Returns:
        str: The token with normalized spacing.
    """
    # Check if the tokenizer adds a leading space to words
    test_token = tokenizer.tokenize('test')[0]
    leading_space = test_token.startswith(' ')

    # Adjust the token based on the tokenizer's behavior
    if leading_space and not token.startswith(' '):
        return ' ' + token
    elif not leading_space and token.startswith(' '):
        return token[1:]
    return token
