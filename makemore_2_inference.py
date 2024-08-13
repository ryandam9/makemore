"""
This is the inference code for the model trained in the previous notebook.
It loads the model weights and generates 20 names using the model.
"""

import torch
import torch.nn.functional as F

itos = {
    0: ".",
    1: "a",
    2: "b",
    3: "c",
    4: "d",
    5: "e",
    6: "f",
    7: "g",
    8: "h",
    9: "i",
    10: "j",
    11: "k",
    12: "l",
    13: "m",
    14: "n",
    15: "o",
    16: "p",
    17: "q",
    18: "r",
    19: "s",
    20: "t",
    21: "u",
    22: "v",
    23: "w",
    24: "x",
    25: "y",
    26: "z",
}

stoi = {s: i for i, s in itos.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 8

model = torch.load("./models/makemore_model.pth", weights_only=False)
model.eval()
model = model.to(device)


def generate_random_names():
    names = list()

    for _ in range(20):
        out = []
        context = [0] * block_size  # initialize with all ...

        while True:
            model_input = torch.tensor([context]).to(device)

            logits = model(model_input)
            probs = F.softmax(logits, dim=1)

            # Sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()

            # shift the context window and track the samples
            context = context[1:] + [ix]

            # If special token, break
            if ix == 0:
                break
            else:
                out.append(ix)

        names.append("".join(itos[i] for i in out))

    print(names)


def generate_names_with_prefix(prefix):
    """
    Prefix: a single character
    """
    names = list()

    for _ in range(20):
        context = [stoi[prefix]] + [0] * (block_size - 1)
        out = [prefix]

        while True:
            model_input = torch.tensor([context]).to(device)

            logits = model(model_input)
            probs = F.softmax(logits, dim=1)

            # Sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()

            # shift the context window and track the samples

            context = context[1:] + [ix]

            # If special token, break
            if ix == 0:
                break
            else:
                out.append(itos[ix])

        names.append("".join(out))

    print(names)


# generate_names_with_prefix("r")
generate_random_names()