import json
from typing import Optional
from nnsight import LanguageModel
import re
import torch
from transformers import AutoModelForCausalLM

import typer

app = typer.Typer()


prompt = "A typical definition of X would be '"


# Define the model names for LLaMA-2, Mistral, and GPT-2
model_names = {
    "llama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
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


class LatentCartographer:
    def __init__(self, model, token_position, cutoff, noken):
        self.model = model
        self.token_position = token_position
        self.cutoff = cutoff
        self.noken = noken

    def loop(self, prompt, nodes, node_id):
        print(f"Looping with prompt: {prompt}")
        with self.model.forward() as runner:
            with runner.invoke(prompt) as _:
                self.model.transformer.wte.output.t[self.token_position] = self.noken
                output = self.model.lm_head.output.t[-1].save()

        cumulative_prob = nodes[node_id]["prob"]

        # Apply softmax, filter out low probability tokens, then get the top k
        probs = torch.softmax(output.value, dim=-1)
        topk = probs.topk(10)
        tokens = [
            (prob.item(), token.item()) for prob, token in zip(topk.values[0], topk.indices[0]) if (cumulative_prob * prob) > self.cutoff
        ]

        for prob, token in tokens:
            word = self.model.tokenizer.decode(token)
            if not validate_word(word):
                print(f"Skipping invalid word: {word}")
                continue

            print(f"{word}: {prob:.4f} ({cumulative_prob * prob:.4f})")

            id = len(nodes) + 1
            nodes[id] = {"token": token, "word": word, "prob": prob * cumulative_prob, "parent": node_id}
            self.loop(prompt + word, nodes, id)


@app.command()
def main(
    word: Optional[str] = typer.Argument(None, help="The word to generate a definition for."),
    model_name: str = "gpt2",
    cutoff: float = 0.0001,
):
    print(f"Generating definition for word: {word} using model: {model_name} with cutoff: {cutoff}")
    if model_name in model_names:
        model_name = model_names[model_name]
    model = LanguageModel(model_name)

    if word is None:
        transformer_model = AutoModelForCausalLM.from_pretrained(model_name)
        embeddings = transformer_model.get_input_embeddings().weight
        embeddings = embeddings.mean(dim=0)
    else:
        with model.forward() as runner:
            with runner.invoke(word) as _:
                embeddings = model.transformer.wte.output.t[0].save()

    tokens = model.tokenizer.encode(prompt)
    try:
        x = model.tokenizer.encode(" X")
        token_position = tokens.index(x[0])
    except ValueError:
        x = model.tokenizer.encode("X")
        token_position = tokens.index(x[0])

    data = {0: {"token": None, "word": prompt, "prob": 1, "parent": None}}
    latent_cartographer = LatentCartographer(model, token_position, cutoff, embeddings)
    latent_cartographer.loop(prompt, data, 0)

    with open("tree.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    app()
