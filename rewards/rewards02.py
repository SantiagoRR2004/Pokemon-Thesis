from rewards.abstractRewards import AbstractRewardFunction
from poke_env.battle import AbstractBattle, Observation


class RewardFunction02(AbstractRewardFunction):

    @staticmethod
    def calculateHPPercent(observation: Observation) -> float:
        """
        Calculate the HP percentage difference between own team and opponent team.

        If the pokemon is not shown, we assume it has full HP.

        Args:
            - observation (Observation): The current observation of the battle.

        Returns:
            - (float): The HP percentage difference.

        """
        hp = 0.0

        # Calculate own team
        for p in observation.team.values():
            hp += p.current_hp_fraction

        # Not shown
        hp += 6 - len(observation.team)

        # Calculate opponent team
        for p in observation.opponent_team.values():
            hp -= p.current_hp_fraction

        # Not shown
        hp -= 6 - len(observation.opponent_team)

        return hp

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
        rewards = []

        # First is throwing the leads
        obs = list(battle.observations.values())[1:]
        currentHP = RewardFunction02.calculateHPPercent(obs[0])

        for o in obs[1:]:
            newHP = RewardFunction02.calculateHPPercent(o)
            rewards.append(newHP - currentHP)
            currentHP = newHP

        # Final reward
        finalReward = 1000 if battle.won else -1000
        rewards.append(finalReward)

        # # Sanity check
        # assert len(rewards) == nSteps, f"Expected {nSteps} rewards, got {len(rewards)}"

        return rewards
