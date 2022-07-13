import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, cuda, emb_dim=768, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cuda = cuda
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.hidden = None

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2).to(self.cuda),
            torch.randn(2, batch_size, self.hidden_dim // 2).to(self.cuda),
        )

    def forward(self, x):
        self.hidden = self.init_hidden(x.shape[0])
        x, self.hidden = self.lstm(x, self.hidden)
        return x
