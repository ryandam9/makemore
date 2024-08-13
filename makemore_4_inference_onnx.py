"""
Inferring from the ONNX model

Ref:
https://pytorch.org/docs/stable/onnx_dynamo.html#a-simple-example
"""

import os

import onnxruntime as ort
import torch
import torch.nn.functional as F

models_location = "./models"
onnx_model = os.path.join(models_location, "makemore.onnx")

ort_session = ort.InferenceSession(onnx_model)

names = list()
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

for _ in range(5):
    out = []
    context = [0] * 8  # initialize with all ...

    while True:
        torch_input = torch.tensor([context])
        onnxruntime_input = {ort_session.get_inputs()[0].name: torch_input.numpy()}
        ort_outs = ort_session.run(None, onnxruntime_input)
        logits = ort_outs[0]

        probs = F.softmax(torch.tensor(logits), dim=1)

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
