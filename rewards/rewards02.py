from rewards.abstractRewards import AbstractRewardFunction
from players.AbstractAIPlayer import RewardData


class RewardFunction02(AbstractRewardFunction):

    @staticmethod
    def calculateHPPercent(observation: RewardData) -> float:
        """
        Calculate the HP percentage difference between own team and opponent team.

        If the pokemon is not shown, we assume it has full HP.

        Args:
            - observation (RewardData): The current observation of the battle.

        Returns:
            - (float): The HP percentage difference.
        """
        hp = 0.0

        # Calculate own team
        for p in observation.team_hp_fractions:
            hp += p

        # Not shown
        hp += 6 - len(observation.team_hp_fractions)

        # Calculate opponent team
        for p in observation.opponent_hp_fractions:
            hp -= p

        # Not shown
        hp -= 6 - len(observation.opponent_hp_fractions)

        return hp

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
        rewards = []

        currentHP = RewardFunction02.calculateHPPercent(battleHistory[0])

        for o in battleHistory[1:]:
            newHP = RewardFunction02.calculateHPPercent(o)
            rewards.append(newHP - currentHP)
            currentHP = newHP

        # Final reward
        finalReward = 1000 if result == 1 else -1000
        rewards.append(finalReward)

        return rewards
