from actors.abstractActor import AbstractActor
from torch import nn


class ActorNetwork04(AbstractActor):

    def generateNetwork(self, player):
        # Layers that slowly get smaller, LeakyReLU activation, and dropout
        return nn.Sequential(
            nn.Linear(player.getNumberOfInputs(), 1024),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(16, player.N_OUTPUTS),
            nn.Softmax(dim=-1),  # Output layer with softmax activation
        )
