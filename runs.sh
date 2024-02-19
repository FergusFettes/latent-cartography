#!/usr/bin/env bash

python demo.py --cutoff 1e-6 --prompt "A typical definition of X would be '"
python demo.py --cutoff 1e-5 --prompt "A typical definition of X would be 'a woman"
python demo.py --cutoff 1e-5 --prompt "A typical definition of X would be 'a man"

python demo.py --model-name gptj --cutoff 1e-6 --prompt "A typical definition of X would be '"
python demo.py --model-name gptj --cutoff 1e-5 --prompt "A typical definition of X would be 'a woman"
python demo.py --model-name gptj --cutoff 1e-5 --prompt "A typical definition of X would be 'a man"

python demo.py --model-name llama --cutoff 1e-6 --prompt "A typical definition of X would be '"
python demo.py --model-name llama --cutoff 1e-5 --prompt "A typical definition of X would be 'a woman"
python demo.py --model-name llama --cutoff 1e-5 --prompt "A typical definition of X would be 'a man"

python demo.py --model-name mistral --cutoff 1e-6 --prompt "A typical definition of X would be '"
python demo.py --model-name mistral --cutoff 1e-5 --prompt "A typical definition of X would be 'a woman"
python demo.py --model-name mistral --cutoff 1e-5 --prompt "A typical definition of X would be 'a man"
