from actors.abstractActor import AbstractActor
from torch import nn


class ActorNetwork03(AbstractActor):

    def generateNetwork(self, player):
        # 4 hidden layers with 256 neurons each, ReLU activation, and dropout
        return nn.Sequential(
            nn.Linear(player.getNumberOfInputs(), 256),
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
            nn.Linear(256, player.N_OUTPUTS),
            nn.Softmax(dim=-1),  # Output layer with softmax activation
        )
