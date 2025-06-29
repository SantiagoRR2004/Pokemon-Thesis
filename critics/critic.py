from players import AIPlayer
from torch import nn


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(AIPlayer.N_F_TOTAL, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output a single value for the state value
        )

    def forward(self, x):
        return self.net(x)
