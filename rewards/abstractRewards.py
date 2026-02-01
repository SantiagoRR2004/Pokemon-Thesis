from poke_env.battle import AbstractBattle
from abc import ABC, abstractmethod


class AbstractRewardFunction(ABC):

    @staticmethod
    @abstractmethod
    def calculateRewards(battle: AbstractBattle) -> list[float]:
        """
        Calculate the rewards for each step in a battle.

        Args:
            - battle: The battle object.

        Returns:
            - List of rewards for each step.
        """
        pass
