from actors.abstractActor import AbstractActor
from torch import nn


class ActorNetwork02(AbstractActor):

    def generateNetwork(self, player):
        return nn.Sequential(
            nn.Linear(player.getNumberOfInputs(), 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, player.N_OUTPUTS),
            nn.Softmax(dim=-1),  # Output layer with softmax activation
        )
