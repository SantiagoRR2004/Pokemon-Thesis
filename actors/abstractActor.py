from players import AbstractAIPlayer
from abc import ABC, abstractmethod
from torch import nn


class AbstractActor(nn.Module, ABC):
    def __init__(self, player: AbstractAIPlayer):
        super(AbstractActor, self).__init__()
        self.net = self.generateNetwork(player)

    @abstractmethod
    def generateNetwork(self, player: AbstractAIPlayer) -> None:
        """
        Here a neural network should be generated that has
        player.N_F_TOTAL inputs and player.N_OUTPUTS outputs.

        Args:
            - player (AbstractAIPlayer): The player for which the network is generated.

        Returns:
            - None
        """
        pass

    def forward(self, x):
        return self.net(x)
