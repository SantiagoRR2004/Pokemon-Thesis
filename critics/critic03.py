from critics.abstractCritic import AbstractCritic
from torch import nn


class CriticNetwork03(AbstractCritic):

    def generateNetwork(self, player):
        # 4 hidden layers with 256 neurons each, ReLU activation, and dropout
        return nn.Sequential(
            nn.Linear(player.N_F_TOTAL, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 1),  # Output a single value for the state value
        )
