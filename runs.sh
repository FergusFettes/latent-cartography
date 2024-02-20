#!/usr/bin/env bash

python cumulative_probability_expansion.py --cutoff 1e-6 --prompt "A typical definition of X would be '"
python cumulative_probability_expansion.py --cutoff 1e-5 --prompt "A typical definition of X would be 'a woman"
python cumulative_probability_expansion.py --cutoff 1e-5 --prompt "A typical definition of X would be 'a man"

python cumulative_probability_expansion.py --model-name gptj --cutoff 1e-6 --prompt "A typical definition of X would be '"
python cumulative_probability_expansion.py --model-name gptj --cutoff 1e-5 --prompt "A typical definition of X would be 'a woman"
python cumulative_probability_expansion.py --model-name gptj --cutoff 1e-5 --prompt "A typical definition of X would be 'a man"

python cumulative_probability_expansion.py --model-name llama --cutoff 1e-6 --prompt "A typical definition of X would be '"
python cumulative_probability_expansion.py --model-name llama --cutoff 1e-5 --prompt "A typical definition of X would be 'a woman"
python cumulative_probability_expansion.py --model-name llama --cutoff 1e-5 --prompt "A typical definition of X would be 'a man"

python cumulative_probability_expansion.py --model-name mistral --cutoff 1e-6 --prompt "A typical definition of X would be '"
python cumulative_probability_expansion.py --model-name mistral --cutoff 1e-5 --prompt "A typical definition of X would be 'a woman"
python cumulative_probability_expansion.py --model-name mistral --cutoff 1e-5 --prompt "A typical definition of X would be 'a man"
