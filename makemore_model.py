import torch
from torch.nn import Embedding, Linear, BatchNorm1d, Tanh
from torch.nn import Module


class FlattenConsecutive(Module):
    def __init__(self, n):
        """
        n - tells how many chars to concatenate in the last dimension of the tensor.

        Given a context of 8 chars: (1, 2, 3, 4, 5, 6, 7, 8)
        if n = 2, then, the output will be:
                (1, 2), (3, 4), (5, 6), (7, 8)
        """
        super().__init__()
        self.n = n

    def forward(self, x):
        # Batch size, Sequence length, no Channels
        B, T, C = x.shape

        # To handle the scenarios where a tensor is
        # contiguous or not.
        x = x.reshape(B, T // self.n, C * self.n)

        # NOTE:
        #   This is to address a special scenario:
        #   Following code changes the shape of a (32, 1, 200) tensor to (32, 200).
        #   why do we need this? If we don't have this, the last layer (logits)
        #   produces a tensor of shape (32, 1, 27) and it will throw an error
        #   when using "F.cross_entropy(logits, Yb)"
        if x.shape[1] == 1:
            x = x.squeeze(1)

        return x


class BatchNorm1dTranspose(Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = BatchNorm1d(num_features)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x


class MakemoreModel(torch.nn.Module):
    def __init__(self, vocab_size, n_dim, n_hidden):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            Embedding(vocab_size, n_dim),
            FlattenConsecutive(2),
            Linear(n_dim * 2, n_hidden, bias=False),
            BatchNorm1dTranspose(n_hidden),
            Tanh(),
            FlattenConsecutive(2),
            Linear(n_hidden * 2, n_hidden, bias=False),
            BatchNorm1dTranspose(n_hidden),
            Tanh(),
            FlattenConsecutive(2),
            Linear(n_hidden * 2, n_hidden, bias=False),
            BatchNorm1d(n_hidden),
            Tanh(),
            Linear(n_hidden, vocab_size),
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits
