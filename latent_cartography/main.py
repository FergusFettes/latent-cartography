import typer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = typer.Typer()


# Define the model names for LLaMA-2, Mistral, and GPT-2
model_names = {
    "llama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "gpt2": "gpt2",
    # "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
}


@app.command()
def app_main(
    model_name: str = typer.Option('gptj', help='Model name or shorthand (e.g. gptj, mistral)'),
    save_directory: str = typer.Option('.', help='Directory where the JSON trees will be saved'),
    topk: int = typer.Option(5, help='Top K probabilities to consider'),
    cutoff: float = typer.Option(0.01, help='Probability cutoff for tree expansion'),
    prompt: str = typer.Option("A typical definition of '_' would be '", help='Base prompt for the model')
):
    """
    Run the latent cartography script to generate and analyze definition trees.
    """
    # Load the model and tokenizer using Auto classes
    full_model_name = model_name  # This should be replaced with a proper mapping if shorthands are used
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    model = AutoModelForCausalLM.from_pretrained(full_model_name)

    # Check if CUDA available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Here you would call the function to generate the definition tree
    # For example: generate_definition_tree(model, tokenizer, topk, cutoff, prompt)

    # Save the results to the specified directory
    # For example: save_results(save_directory, results)

    typer.echo(f"Definition tree generation complete. Results saved to {save_directory}")

if __name__ == "__main__":
    app()
