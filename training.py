import randomTeams.randomTeam as randomTeam
from players import AIPlayer
from actors import ActorNetwork01
from critics import CriticNetwork01
from poke_env.player import RandomPlayer
import torch
import torch.nn as nn
import torch.optim as optim
import serverControl
import metricsLogger
import asyncio
import time
import os


async def main(
    *,
    actor: nn.Module,
    nTeams: int = float("inf"),
    critic: nn.Module = None,
    nEpisodes: int = 64,
    fileName: str = None,
) -> None:
    """
    Train an actor-critic model for a given number of episodes.

    Args:
        - actor (nn.Module): The actor network to be trained.
        - nTeams (int): Number of teams to use for training. If float("inf"), random teams will be used.
        - critic (nn.Module, optional): The critic network to be trained. If None, only the actor will be trained.
        - nEpisodes (int): Number of episodes to run for training.
        - fileName (str, optional): The name of the file to save the metrics. If None, a default name will be used.

    Returns:
        - None
    """
    p = serverControl.startServer()

    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    if critic:
        criticOptimizer = optim.Adam(critic.parameters(), lr=1e-3)

    nEpochs = 1000

    victoryPercentage = []
    actorLosses = []
    averageRewards = []
    if critic:
        criticLosses = []
        averageCriticRewards = []
    nTurns = []

    start = time.time()

    for epoch in range(nEpochs):

        if nTeams == float("inf"):
            # We create the AI player
            player = AIPlayer(
                battle_format="gen9randombattle",
                network=actor,
                critic=critic,
                max_concurrent_battles=nEpisodes,
            )

            # We create another random player
            second_player = RandomPlayer(
                battle_format="gen9randombattle",
                max_concurrent_battles=nEpisodes,
            )
        else:

            # We create the AI player
            player = AIPlayer(
                battle_format="gen9purehackmons",
                team=randomTeam.selectRandomTeam(nTeams),
                network=actor,
                critic=critic,
                max_concurrent_battles=nEpisodes,
            )

            # We create another random player
            second_player = RandomPlayer(
                battle_format="gen9purehackmons",
                team=randomTeam.selectRandomTeam(nTeams),
                max_concurrent_battles=nEpisodes,
            )

        # Reset the player for new Episodes
        player.reset()

        # Reset the battle counters
        player.reset_battles()
        second_player.reset_battles()

        # Run n_battles (Episodes)
        await player.battle_against(second_player, n_battles=nEpisodes)

        actorLoss = 0
        averageRewardsEpoch = 0
        if critic:
            criticLoss = 0
            averageCriticRewardsEpoch = 0
        gamma = 0.99  # discount factor (the far future is very important)

        for battle in player.battles.values():
            nSteps = battle.turn
            actorLossBattle = 0
            if critic:
                criticLossBattle = 0
            finalReward = 1000 if battle.won else -1000

            # Reward sequence
            rewards = [1] * (nSteps - 1) + [finalReward]

            # Compute discounted returns
            discountedRewards = []
            cumulative = 0
            for r in reversed(rewards):
                cumulative = r + gamma * cumulative
                discountedRewards.insert(0, cumulative)

            averageRewardsEpoch += (
                torch.tensor(discountedRewards, dtype=torch.float32).mean().item()
            )
            if critic:
                averageCriticRewardsEpoch += (
                    torch.stack(player.values[battle.battle_tag]).mean().item()
                )

            # Calculate the loss using actor-critic
            if critic:
                for log_prob, G, V in zip(
                    player.log_probs[battle.battle_tag],
                    discountedRewards,
                    player.values[battle.battle_tag],
                ):
                    advantage = G - V.item()
                    actorLossBattle += -log_prob * advantage

                for G, V in zip(discountedRewards, player.values[battle.battle_tag]):
                    criticLossBattle += (V - G).pow(2)
            # Calculate the loss using only actor
            else:
                for log_prob, G in zip(
                    player.log_probs[battle.battle_tag], discountedRewards
                ):
                    actorLossBattle += -log_prob * G

            # Normalize the loss by the episodes
            actorLoss += actorLossBattle / nSteps
            if critic:
                criticLoss += criticLossBattle / nSteps

        # Normalize the loss by the number of battles
        actorLoss /= len(player.battles)
        if critic:
            criticLoss /= len(player.battles)

        # Backpropagation step
        optimizer.zero_grad()
        actorLoss.backward()
        optimizer.step()

        # Update the critic network
        if critic:
            criticOptimizer.zero_grad()
            criticLoss.backward()
            criticOptimizer.step()

        percentage = player.n_won_battles / player.n_finished_battles

        # We can now print the results of the battles
        print(f"{epoch+1:0{len(str(nEpochs))}d}/{nEpochs}", end=" ", flush=True)
        print(
            time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    (time.time() - start) * (nEpochs - (epoch + 1)) / (epoch + 1)
                ),
            ),
            end=" ",
            flush=True,
        )
        print(
            f"{player.username} won {percentage*100:.2f}% of battles.",
            end=" ",
            flush=True,
        )
        print()

        victoryPercentage.append(percentage)
        actorLosses.append(actorLoss.item())
        averageRewards.append(averageRewardsEpoch / len(player.battles))
        if critic:
            criticLosses.append(criticLoss.item())
            averageCriticRewards.append(averageCriticRewardsEpoch / len(player.battles))
        nTurns.append(
            sum(battle.turn for battle in player.battles.values()) / len(player.battles)
        )

        if (epoch + 1) % 100 == 0 and (epoch + 1) != nEpochs:
            serverControl.endProcess(p)
            p.wait()
            p = serverControl.startServer()

    # Save the metrics
    kwargs = {
        "victoryPercentage": victoryPercentage,
        "actorLosses": actorLosses,
        "averageRewards": averageRewards,
        "nTurns": nTurns,
    }

    if critic:
        kwargs["criticLosses"] = criticLosses
        kwargs["averageCriticRewards"] = averageCriticRewards

    if fileName is not None:
        kwargs["fileName"] = fileName

    metricsLogger.saveData(**kwargs)


if __name__ == "__main__":

    if os.name == "nt":
        # For Windows
        os.system("cls")
    else:
        # For Linux/macOS
        os.system("clear")

    import ignorePokeEnvProblems

    asyncio.run(
        main(
            actor=ActorNetwork01(AIPlayer),
            critic=CriticNetwork01(AIPlayer),
            nTeams=float("inf"),
        )
    )
