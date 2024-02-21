import time
import json
from typing import Optional
from nnsight import LanguageModel
import re
import torch
from transformers import AutoModelForCausalLM

from tqdm import tqdm
import typer

app = typer.Typer()


# Define the model names for LLaMA-2, Mistral, and GPT-2
model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    # "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
}


def validate_word(word):
    word = word.strip()
    if not word:
        return False
    if not re.match(r"^[a-zA-Z']+$", word):
        return False
    return True


def get_model_specifics(model_name):
    """
    Get the model specific attributes.
    The following works for gpt2, llama2 and mistral models.
    """
    if "gpt" in model_name:
        return "transformer", "h", "wte"
    if "mamba" in model_name:
        return "backbone", "layers", "embed_tokens"
    return "model", "layers", "embed_tokens"


def top_p_filtering(probs, top_p=0.9):
    """Filter a distribution of probabilities using top-p filtering"""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with a cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Create a mask for all the indices to remove
    mask = torch.ones_like(probs).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    # Filter out low probability tokens by setting them to 0
    filtered_probs = probs * mask
    # Renormalize probabilities
    filtered_probs = filtered_probs / filtered_probs.sum()
    return filtered_probs


class LatentCartographer:
    def __init__(self, model, model_name, token_position, cutoff, noken):
        self.model = model
        self.token_position = token_position
        self.cutoff = cutoff
        self.noken = noken
        self.model_specifics = get_model_specifics(model_name)
        self.nodes = {}
        self.progress_bar = tqdm(desc="Processing", unit="iter")

    def loop(self, prompt_tokens, node_id):
        self.progress_bar.update(1)
        prompt = self.model.tokenizer.decode(prompt_tokens)
        with self.model.forward() as runner:
            with runner.invoke(prompt) as _:
                getattr(getattr(self.model, self.model_specifics[0]), self.model_specifics[2]).output.t[self.token_position] = self.noken
                output_logits = self.model.lm_head.output.t[-1].save()

        # Get the cumulative probability from the parent node
        parent_cumulative_prob = self.nodes[node_id]["prob"]

        # Apply softmax to convert logits into probabilities and scale by parent's cumulative probability
        scaled_probs = torch.nn.functional.softmax(output_logits.value, dim=-1) * parent_cumulative_prob

        # Apply top-p filtering to get the truncated distribution of probabilities
        filtered_probs = top_p_filtering(scaled_probs, top_p=self.cutoff)

        # Sample from the truncated distribution defined by the filtered probabilities
        tokens_and_probs = [
            (prob.item(), token_idx.item())
            for token_idx, prob in enumerate(filtered_probs)
            if prob.item() > 0
        ]

        for prob, token in tokens_and_probs:
            cumulative_prob = prob * parent_cumulative_prob  # Update cumulative probability
            word = self.model.tokenizer.decode([token])  # Ensure token is a list for decoding
            if not validate_word(word):
                tqdm.write(f"Skipping invalid word: {word}")
                continue

            tqdm.write(f"prompt: {prompt} -> {word}:\tprob: {prob:.4f}, cumulative: {cumulative_prob:.4f}")

            id = len(self.nodes) + 1
            self.nodes[id] = {"token": token, "word": word, "prob": cumulative_prob, "parent": node_id}
            self.loop(prompt_tokens + [token], id)


@app.command()
def main(
    word: Optional[str] = typer.Argument(None, help="The word to generate a definition for."),
    model_name: str = "gpt2",
    cutoff: float = 0.0001,
    prompt: str = typer.Option("A typical definition of X would be '", help="Must contain X, which will be replaced with the word"),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating definition for word: {word} using model: {model_name} with cutoff: {cutoff}")
    if model_name in model_names:
        model_name = model_names[model_name]
    model = LanguageModel(model_name, device_map=device)

    if word is None:
        transformer_model = AutoModelForCausalLM.from_pretrained(model_name)
        embeddings = transformer_model.get_input_embeddings().weight
        embeddings = embeddings.mean(dim=0)
        word = "noken"
    else:
        model_specifics = get_model_specifics(model_name)

        with model.forward() as runner:
            with runner.invoke(word) as _:
                embeddings = getattr(getattr(model, model_specifics[0]), model_specifics[2]).output.t[0].save()

    tokens = model.tokenizer.encode(prompt)
    try:
        x = model.tokenizer.encode(" X")
        token_position = tokens.index(x[0])
    except ValueError:
        x = model.tokenizer.encode("X")
        token_position = tokens.index(x[0])

    tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
    latent_cartographer = LatentCartographer(model, model_name, token_position, cutoff, embeddings)
    latent_cartographer.nodes = {0: {"token": None, "word": prompt, "prob": 1, "parent": None}}
    start = time.time()
    latent_cartographer.loop(tokens, 0)

    print(f"Elapsed time: {time.time() - start:.2f}s. Nodes: {len(latent_cartographer.nodes)}")

    model_name = model_name.replace("/", "-")
    filename = f"{model_name}_{word}_{cutoff}_{prompt}.json"
    print(f"Saving nodes to {filename}")
    with open(filename, "w") as f:
        json.dump(latent_cartographer.nodes, f, indent=2)


if __name__ == "__main__":
    app()
