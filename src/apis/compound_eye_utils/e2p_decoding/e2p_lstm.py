import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Define the LSTM model
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers=4, bidirectional=True, use_attention=False):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=1,  # Input is 1-channel
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Attention mechanism (currently not used)
        if use_attention:
            self.attention = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)  # Output is 1-channel

    def forward(self, x):
        # x shape: (batch_size, seq_length, 1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size * num_directions)

        # Apply fully connected layer to each time step
        out = self.fc(lstm_out)  # (batch_size, seq_length, 1)

        out = torch.sigmoid(out)

        return out
# Define the LSTM model
# class LSTMDecoder(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMDecoder, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # x shape: (batch_size, seq_length, input_size)
#         lstm_out, _ = self.lstm(x)
#         # Get the last time step output
#         out = self.fc(lstm_out)
#         return out