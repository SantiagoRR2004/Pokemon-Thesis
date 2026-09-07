from rewards.abstractRewards import AbstractRewardFunction
from players.AbstractAIPlayer import RewardData


class RewardFunction01(AbstractRewardFunction):

    @staticmethod
    def calculateRewards(battleHistory: list[RewardData], result: int) -> list[float]:
        """
        Calculate the rewards for each step in a battle.

        For this reward function, we give a reward of +1 for each turn survived
        and a final reward of +1000 for winning or -1000 for losing.

        Args:
            - battleHistory (list[RewardData]): The history of the battle.
            - result (int): The result of the battle (1 for win, -1 for loss, 0 for draw).

        Returns:
            - List of rewards for each step.
        """
        nSteps = len(battleHistory)
        finalReward = 1000 if result == 1 else -1000

        # Reward sequence
        rewards = [1] * (nSteps - 1) + [finalReward]

        return rewards
