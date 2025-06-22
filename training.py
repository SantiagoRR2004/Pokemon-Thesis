from MyAIPlayer import AIPlayer
from poke_env.player import RandomPlayer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import serverControl
import metricsLogger
import asyncio
import os


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(AIPlayer.N_F_TOTAL, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, AIPlayer.N_OUTPUTS),
            nn.Softmax(dim=-1),  # Output layer with softmax activation
        )

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(AIPlayer.N_F_TOTAL, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output a single value for the state value
        )

    def forward(self, x):
        return self.net(x)


async def main():
    serverControl.startServer()

    with open("randomTeam1.txt", "r") as f:
        random_team1 = f.read()

    with open("randomTeam2.txt", "r") as f:
        random_team2 = f.read()

    model = NeuralNetwork()
    critic = CriticNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criticOptimizer = optim.Adam(critic.parameters(), lr=1e-3)
    nEpisodes = 64
    nEpochs = 40

    # We create the AI player
    player = AIPlayer(
        battle_format="gen9anythinggoes",
        team=random_team1,
        network=model,
        critic=critic,
        max_concurrent_battles=nEpisodes,
    )

    # We create another random player
    second_player = RandomPlayer(
        battle_format="gen9anythinggoes",
        team=random_team2,
        max_concurrent_battles=nEpisodes,
    )

    victoryPercentage = []
    actorLosses = []
    criticLosses = []
    averageRewards = []
    averageCriticRewards = []
    nTurns = []

    for epoch in range(nEpochs):

        # Reset the player for new Episodes
        player.reset()

        # Reset the battle counters
        player.reset_battles()
        second_player.reset_battles()

        # Run n_battles (Episodes)
        await player.battle_against(second_player, n_battles=nEpisodes)

        actorLoss = 0
        criticLoss = 0
        averageRewardsEpoch = 0
        averageCriticRewardsEpoch = 0
        gamma = 0.99  # discount factor (the far future is very important)

        for battle in player.battles.values():
            nSteps = battle.turn
            actorLossBattle = 0
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
            averageCriticRewardsEpoch += (
                torch.stack(player.values[battle.battle_tag]).mean().item()
            )

            # Calculate the loss using actor-critic
            for log_prob, G, V in zip(
                player.log_probs[battle.battle_tag],
                discountedRewards,
                player.values[battle.battle_tag],
            ):
                advantage = G - V.item()
                actorLossBattle += -log_prob * advantage

            for G, V in zip(discountedRewards, player.values[battle.battle_tag]):
                criticLossBattle += (V - G).pow(2)

            # Normalize the loss by the episodes
            actorLoss += actorLossBattle / nSteps
            criticLoss += criticLossBattle / nSteps

        # Normalize the loss by the number of battles
        actorLoss /= len(player.battles)
        criticLoss /= len(player.battles)

        # Backpropagation step
        optimizer.zero_grad()
        actorLoss.backward()
        optimizer.step()

        # Update the critic network
        criticOptimizer.zero_grad()
        criticLoss.backward()
        criticOptimizer.step()

        percentage = player.n_won_battles / player.n_finished_battles

        # We can now print the results of the battles
        print(f"{epoch+1:0{len(str(nEpochs))}d}/{nEpochs}", end=" ")
        print(
            f"Player {player.username} won {percentage*100:.2f}% of battles.", end=" "
        )
        print()

        victoryPercentage.append(percentage)
        actorLosses.append(actorLoss.item())
        criticLosses.append(criticLoss.item())
        averageRewards.append(averageRewardsEpoch / len(player.battles))
        averageCriticRewards.append(averageCriticRewardsEpoch / len(player.battles))
        nTurns.append(
            sum(battle.turn for battle in player.battles.values()) / len(player.battles)
        )

    metricsLogger.saveData(
        victoryPercentage=victoryPercentage,
        actorLosses=actorLosses,
        criticLosses=criticLosses,
        averageRewards=averageRewards,
        averageCriticRewards=averageCriticRewards,
        nTurns=nTurns,
    )


if __name__ == "__main__":

    if os.name == "nt":
        # For Windows
        os.system("cls")
    else:
        # For Linux/macOS
        os.system("clear")

    import ignorePokeEnvProblems

    asyncio.run(main())
