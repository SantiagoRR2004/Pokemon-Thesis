from rewards.abstractRewards import AbstractRewardFunction
from poke_env.battle import AbstractBattle


class RewardFunction01(AbstractRewardFunction):

    @staticmethod
    def calculateRewards(battle: AbstractBattle) -> list[float]:
        """
        Calculate the rewards for each step in a battle.

        For this reward function, we give a reward of +1 for each turn survived
        and a final reward of +1000 for winning or -1000 for losing.

        Args:
            - battle: The battle object.

        Returns:
            - List of rewards for each step.
        """
        nSteps = battle.turn
        finalReward = 1000 if battle.won else -1000

        # Reward sequence
        rewards = [1] * (nSteps - 1) + [finalReward]

        return rewards
