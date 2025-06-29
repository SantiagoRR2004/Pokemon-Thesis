from critics.abstractCritic import AbstractCritic
from torch import nn


class CriticNetwork(AbstractCritic):

    def generateNetwork(self, player):
        return nn.Sequential(
            nn.Linear(player.N_F_TOTAL, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output a single value for the state value
        )
