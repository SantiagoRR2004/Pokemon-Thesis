from actors.abstractActor import AbstractActor
from torch import nn


class ActorNetwork01(AbstractActor):

    def generateNetwork(self, player):
        # Most basic actor network with 1 hidden layer
        return nn.Sequential(
            nn.Linear(player.getNumberOfInputs(), player.N_OUTPUTS),
        )
