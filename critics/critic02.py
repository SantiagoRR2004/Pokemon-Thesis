from critics.abstractCritic import AbstractCritic
from torch import nn


class CriticNetwork02(AbstractCritic):

    def generateNetwork(self, player):
        # Most basic critic network with one linear layer
        return nn.Sequential(nn.Linear(player.getNumberOfInputs(), 1))
