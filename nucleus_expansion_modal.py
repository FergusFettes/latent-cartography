import time
import json
import re
import torch
from transformers import AutoModelForCausalLM

from modal import Stub, gpu, method
from modal_image_gptj import image
from tqdm import tqdm


stub = Stub(image=image, name="nnsight")

# Define the model names for LLaMA-2, Mistral, and GPT-2
model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    # "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
    "gemma": "google/gemma-7b",
    "gemma2": "google/gemma-2b",
}


@stub.cls(
    gpu=gpu.A100(memory=40, count=1),
    timeout=60 * 10,
    container_idle_timeout=60 * 5,
)
class LatentCartographer:
    def setup_model(self, model_name, cutoff, word, prompt, use_noken=True):
        from nnsight import LanguageModel
        self.model = LanguageModel(model_name, device_map="cuda")
        self.model_name = model_name
        self.word = word
        self.cutoff = cutoff
        self.prompt = prompt
        self.model_specifics = LatentCartographer.get_model_specifics(model_name)
        self.progress_bar = tqdm(desc="Processing", unit="iter")
        self.nodes = {0: {"token": None, "word": self.prompt, "prob": 1, "parent": None}}
        self.use_noken = use_noken

    @staticmethod
    def validate_word(word):
        word = word.strip()
        if not word:
            return False
        if not re.match(r"^[a-zA-Z']+$", word):
            return False
        return True

    @staticmethod
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

    @staticmethod
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

    def loop(self, prompt_tokens, node_id):
        self.progress_bar.update(1)
        prompt = self.model.tokenizer.decode(prompt_tokens)
        with self.model.forward() as runner:
            with runner.invoke(prompt) as _:
                if self.use_noken:
                    getattr(getattr(
                        self.model, self.model_specifics[0]
                    ), self.model_specifics[2]).output.t[self.token_position] = self.embeddings
                output = self.model.lm_head.output.t[-1].save()

        cumulative_prob = self.nodes[node_id]["prob"]

        # Apply softmax, filter out low probability tokens, then get the top k
        probs = torch.softmax(output.value, dim=-1)
        topk = probs.topk(10)
        tokens = [
            (prob.item(), token.item()) for prob, token in zip(topk.values[0], topk.indices[0]) if (cumulative_prob * prob) > self.cutoff
        ]

        for prob, token in tokens:
            word = self.model.tokenizer.decode(token)
            if not LatentCartographer.validate_word(word):
                tqdm.write(f"Skipping invalid word: {word}")
                continue

            tqdm.write(f"prompt: {prompt} -> {word}:\t{prob:.4f}\t({cumulative_prob * prob:.2e})")

            id = len(self.nodes) + 1
            self.nodes[id] = {"token": token, "word": word, "prob": prob * cumulative_prob, "parent": node_id}
            self.loop(prompt_tokens + [token], id)

    def get_word(self, word):
        if word is None:
            transformer_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            embeddings = transformer_model.get_input_embeddings().weight
            self.embeddings = embeddings.mean(dim=0)
            self.word = "noken"
        else:
            with self.model.forward() as runner:
                with runner.invoke(word) as _:
                    self.embeddings = getattr(getattr(self.model, self.model_specifics[0]), self.model_specifics[2]).output.t[0].save()

    def get_tokens(self, prompt):
        tokens = self.model.tokenizer.encode(prompt)
        if self.use_noken:
            try:
                x = self.model.tokenizer.encode(" X")
                self.token_position = tokens.index(x[0])
            except ValueError:
                x = self.model.tokenizer.encode("X")
                self.token_position = tokens.index(x[0])

        self.tokens = self.model.tokenizer.encode(prompt, add_special_tokens=False)

    @method()
    def run(self, model_name, cutoff, word, prompt, use_noken):
        self.setup_model(model_name, cutoff, word, prompt, use_noken)
        print(f"Generating definition for word: {self.word} using model: {self.model_name} with cutoff: {self.cutoff}")
        if self.use_noken:
            self.get_word(self.word)
        self.get_tokens(self.prompt)
        print(f"Running loop with tokens: {self.tokens}")
        self.loop(self.tokens, 0)
        return self.nodes


@stub.local_entrypoint()
def main(
    word: str = None,
    model_name: str = "gptj",
    cutoff: float = 0.0001,
    prompt: str = "A typical definition of X would be '",
    use_noken: bool = True
):
    print(f"Generating definition for word: {word} using model: {model_name} with cutoff: {cutoff}")
    if model_name in model_names:
        model_name = model_names[model_name]

    latent_cartographer = LatentCartographer()
    start = time.time()
    nodes = latent_cartographer.run.remote(model_name, cutoff, word, prompt, use_noken)

    print(f"Elapsed time: {time.time() - start:.2f}s. Nodes: {len(nodes)}")

    model_name = model_name.replace("/", "-")
    filename = f"{model_name}_{word}_{cutoff}_{prompt}.json"
    print(f"Saving nodes to {filename}")
    with open(filename, "w") as f:
        json.dump(nodes, f, indent=2)
