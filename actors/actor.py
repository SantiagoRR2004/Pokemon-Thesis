from players import AIPlayer
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(AIPlayer.N_F_TOTAL, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, AIPlayer.N_OUTPUTS),
            nn.Softmax(dim=-1),  # Output layer with softmax activation
        )

    def forward(self, x):
        return self.net(x)
