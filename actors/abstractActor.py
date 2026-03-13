from players import AbstractAIPlayer
from abc import ABC, abstractmethod
from torch import nn
import torch


class AbstractActor(nn.Module, ABC):
    def __init__(self, player: AbstractAIPlayer):
        super(AbstractActor, self).__init__()
        self.net = self.generateNetwork(player)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(self.device))
