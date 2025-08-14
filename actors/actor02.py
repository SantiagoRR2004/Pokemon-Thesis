from actors.abstractActor import AbstractActor
from torch import nn


class ActorNetwork02(AbstractActor):

    def generateNetwork(self, player):
        # Most basic actor network with 1 hidden layer
        return nn.Sequential(
            nn.Linear(player.getNumberOfInputs(), player.N_OUTPUTS),
            nn.Softmax(dim=-1),  # Output layer with softmax activation
        )
