from nnsight import LanguageModel
import re
import torch


prompt = "A typical definition of X would be '"


model = LanguageModel("gpt2")


def validate_word(word):
    word = word.strip()
    if not word:
        return False
    if not re.match(r"^[a-zA-Z']+$", word):
        return False
    return True


tokens = model.tokenizer.encode(prompt)
try:
    x = model.tokenizer.encode(" X")
    token_position = tokens.index(x[0])
except ValueError:
    x = model.tokenizer.encode("X")
    token_position = tokens.index(x[0])

testword = "apple"
with model.forward() as runner:
    with runner.invoke(testword) as invoker:
        testword_embeddings = model.transformer.wte.output.t[0].save()


def loop(prompt, nodes, node_id):
    print(f"Looping with prompt: {prompt}")
    with model.forward() as runner:
        with runner.invoke(prompt) as _:
            model.transformer.wte.output.t[token_position] = testword_embeddings
            output = model.lm_head.output.t[-1].save()

    cumulative_prob = nodes[node_id]["prob"]

    # Apply softmax, filter out low probability tokens, then get the top k
    cutoff = 0.0001
    probs = torch.softmax(output.value, dim=-1)
    topk = probs.topk(10)
    tokens = [(prob.item(), token.item()) for prob, token in zip(topk.values[0], topk.indices[0]) if (cumulative_prob * prob) > cutoff]

    for prob, token in tokens:
        word = model.tokenizer.decode(token)
        if not validate_word(word):
            print(f"Skipping invalid word: {word}")
            continue

        print(f"{word}: {prob:.4f} ({cumulative_prob * prob:.4f})")

        id = len(nodes) + 1
        nodes[id] = {"token": token, "word": word, "prob": prob * cumulative_prob, "parent": node_id}
        loop(prompt + word, nodes, id)


data = {0: {"token": None, "word": prompt, "prob": 1, "parent": None}}
loop(prompt, data, 0)


import json
with open("tree.json", "w") as f:
    json.dump(data, f, indent=2)
