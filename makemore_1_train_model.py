"""
makemore
  - Train a character-level neural network to generate new names.

Source:
https://www.youtube.com/watch?v=t3YJ5hKiMQ0
"""

import random

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, BatchNorm1d
from torch.utils.data import Dataset, DataLoader

from makemore_model import MakemoreModel


class NameDataset(Dataset):
    def __init__(self, words, block_size, stoi):
        """
        Prepares training data

        Args:
            words: list of words
            block_size: size of the context window
            stoi: string to index

        Returns:
            None
        """
        X, Y = [], []

        for w in words:
            context = [0] * block_size
            for ch in w + ".":
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]  # crop and append

        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.Y.shape[0]


def verify_batches(data_loader, itos):
    """
    Verify the data loader by printing a few samples.
    """

    print("\nSample batch data from the data loader")
    print("-" * 120)

    for epoch in range(1):
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= 3:
                break

            print(" Batch index:", batch_idx, end="")
            print(" | Batch size:", y.shape[0], end="")
            print(" | x shape:", x.shape, end="")
            print(" | y shape:", y.shape)

    # print a batch of data
    for ex, label in zip(x, y):
        for ch in ex:
            print(itos[ch.item()], end="")

        print(" => ", itos[label.item()])


def test_layer_shapes(model, data_loader):
    """
    This is to mimic forward pass during training where all the batches
    are passed through the model.

    To verify the shapes of output from each layer.
    """

    print("\noutput shapes from each layer")
    print("-" * 120)

    for epoch in range(1):
        for batch_idx, (x, y) in enumerate(data_loader):
            print("x:", x.shape)

            test_x = x
            for i in range(len(model.all_layers)):
                test_x = model.all_layers[i](test_x)
                print(
                    f"Layer {i}: {model.all_layers[i].__class__.__name__} ",
                    test_x.shape,
                )

            break


def test_single_batch(model):
    """
    This is to mimic forward pass at Inference time where a single batch
    is passed through the model.
    """
    print("\nShapes of output from each layer for a single batch")
    print("-" * 120)

    model.eval()

    with torch.no_grad():
        context = [0] * block_size
        x = torch.tensor([context])

        for i in range(len(model.all_layers)):
            x = model.all_layers[i](x)
            print(f"Layer {i}: {model.all_layers[i].__class__.__name__} ", x.shape)


def print_model_params(model):
    """
    Print the model parameters and gradients
    """
    print("\nParameters in each layer")
    print("-" * 120)

    total_params = 0
    i = 0
    for name, layer in model.named_modules():
        if isinstance(layer, (Embedding, Linear, BatchNorm1d)):
            i += 1
            for param_name, param in layer.named_parameters():
                param_count = param.numel()
                total_params += param_count
                print(
                    f"{i:<2} {layer.__class__.__name__:<15}: "
                    f"Shape: {str(param.shape):<25} "
                    f"Parameters: {str(param_count):<8} "
                    f"Gradient: {'Enabled' if param.requires_grad else 'Disabled'}"
                )

    print(f"\nTotal parameters: {total_params}")


def compute_total_loss(model, dataloader, device=torch.device("cpu")):
    """
    Compute the total loss of the model on the given dataset.
    """
    model = model.to(device)
    model.eval()

    loss = 0.0
    examples = 0.0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
            batch_loss = F.cross_entropy(logits, labels, reduction="sum")

    loss += batch_loss.item()
    examples += logits.shape[0]

    return loss / examples


@torch.no_grad()
def generate(model, block_size, itos, device, how_many=20):
    """ """
    names = list()

    for _ in range(how_many):
        out = []
        context = [0] * block_size  # initialize with all ...

        while True:
            logits = model(torch.tensor([context]).to(device))
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

    return names


# Hyperparameters
block_size = 8  # context window
n_dim = 24  # Dimension of each character embedding
n_hidden = 128  # Hidden units in linear layers
num_epochs = 50
batch_size = 32
no_epochs = 50

# Read the names file
words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))

# String to Index
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0

# Index to String
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

# Shuffle the words
random.seed(42)
random.shuffle(words)

# Data loaders
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

train_dataset = NameDataset(words[:n1], block_size, stoi)
test_dataset = NameDataset(words[n1:n2], block_size, stoi)
val_dataset = NameDataset(words[n2:], block_size, stoi)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

verify_batches(train_loader, itos)

# Model
model = MakemoreModel(vocab_size, n_dim, n_hidden)

print("\nModel:")
print("-" * 120)
print(model)

test_layer_shapes(model, train_loader)
test_single_batch(model)
print_model_params(model)

# Training
print("Torch CUDA available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

training_loss = []
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(num_epochs):
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)

        # Skip the batch if the size is not as expected
        if labels.shape[0] != batch_size:
            continue

        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        # Learning rate decay
        lr = 0.1 if epoch < 25 else 0.01

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        training_loss.append(loss.log10().item())

    print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}" f" | Train Batch Loss: {loss:.4f}")

# Loss on training data
print(
    f"\nLoss on training set: {compute_total_loss(model, train_loader, device=device)}"
)

# Loss on validation data
print(f"Loss on validation set: {compute_total_loss(model, val_loader, device=device)}")

# Loss on test data
print(f"Loss on test set: {compute_total_loss(model, test_loader, device=device)}")

# Generate names
names = generate(model, block_size, itos, device, how_many=2)

print("\nGenerated names:")
for name in names:
    print(name)

# Save the model
model_path = "./models/makemore_model.pth"
torch.save(model, model_path)
print(f"\nModel saved to {model_path}")
