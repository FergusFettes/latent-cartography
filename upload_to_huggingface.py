import typer
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, Repository
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
app = typer.Typer()


def upload_dataset_to_huggingface(dataset, model_name, word, cutoff, prompt):
    # Load the HuggingFace token from the environment
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    username = os.getenv("HUGGINGFACE_USERNAME")
    if not huggingface_token:
        raise ValueError("HuggingFace token not found. Please set it in your environment variables.")

    # Convert the dataset to a HuggingFace DatasetDict
    def gen():
        for key, value in dataset.items():
            yield value
    hf_dataset = Dataset.from_generator(gen)

    hf_dataset.info.description = f"Dataset for {model_name} embedding centroid expansion. The dataset was created using a cutoff of {cutoff} and the prompt '{prompt}'."
    dataset_dict = DatasetDict({'train': hf_dataset})

    # Only take alphanumeric characters from the prompt
    prompt = re.sub(r'\W+', '', prompt)

    dataset_repository = f"{username}/{model_name}_{word}_{cutoff}_{prompt}"
    repo_url = HfApi(token=huggingface_token).create_repo(dataset_repository, exist_ok=True, repo_type="dataset")
    repo_local_path = f"./{model_name}_{word}_{cutoff}_{prompt}"

    # Clone repository and copy dataset files to it
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=huggingface_token)

    # Move the dataset to the repository as test.json
    json.dump(dataset, open(os.path.join(repo_local_path, 'train.json'), 'w'))

    # Save dataset to the repository
    dataset_dict.save_to_disk(repo_local_path)

    # # Push the dataset to the HuggingFace Hub
    # hf_dataset.push_to_hub(f"{username}/{model_name}_{word}_{cutoff}_{prompt}", token=huggingface_token)
    with open(os.path.join(repo_local_path, 'README.md'), 'w') as readme_file:
        readme_file.write(hf_dataset.info.description)

    # Push changes to the HuggingFace Hub
    repo.push_to_hub(commit_message="Updating dataset from upload_to_huggingface.py")


def extract_metadata_from_filename(filename):
    pattern = re.compile(r"^(.+)_(.+)_(.+)_(.+)\.json$")
    match = pattern.match(filename)
    if not match:
        raise ValueError("Filename does not match the expected pattern.")
    return match.groups()


@app.command()
def main(filename: str):
    try:
        model_name, word, cutoff, prompt = extract_metadata_from_filename(filename)
        print(model_name, word, cutoff, prompt)
        with open(filename, 'r') as file:
            dataset = json.load(file)
        upload_dataset_to_huggingface(dataset, model_name, word, cutoff, prompt)
        typer.echo(f"Dataset from {filename} uploaded successfully.")
    except Exception as e:
        typer.echo(f"An error occurred: {e}")


if __name__ == "__main__":
    typer.run(main)
