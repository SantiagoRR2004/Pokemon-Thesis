from players.AbstractAIPlayer import RewardData
from abc import ABC, abstractmethod


class AbstractRewardFunction(ABC):

    @staticmethod
    @abstractmethod
    def calculateRewards(battleHistory: list[RewardData], result: int) -> list[float]:
        """
        Calculate the rewards for each step in a battle.

        Args:
            - battleHistory (list[RewardData]): The history of the battle.
            - result (int): The result of the battle (1 for win, -1 for loss, 0 for draw).

        Returns:
            - List of rewards for each step.
        """
        pass
