from MyAIPlayer import AIPlayer
from poke_env.player import RandomPlayer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import serverControl
import asyncio
import os


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(AIPlayer.N_F_TOTAL, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, AIPlayer.N_OUTPUTS)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=-1)  # Apply softmax to get probabilities
        return x


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(AIPlayer.N_F_TOTAL, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output a single value for the state value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
    nEpisodes = 32
    nBatches = 10

    # We create a random player
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

    for batch in range(nBatches):

        # Reset the player for new Episodes
        player.reset()

        # Reset the battle counters
        player.reset_battles()
        second_player.reset_battles()

        # Run n_battles (Episodes)
        await player.battle_against(second_player, n_battles=nEpisodes)

        actorLoss = 0
        criticLoss = 0
        averageRewardsBatch = 0
        averageCriticRewardsBatch = 0
        gamma = 0.99  # discount factor (the far future is very important)

        for battle in player.battles.values():
            nEpisodes = battle.turn
            finalReward = 1000 if battle.won else -1000

            # Reward sequence
            rewards = [1] * (nEpisodes - 1) + [finalReward]

            # Compute discounted returns
            discountedRewards = []
            cumulative = 0
            for r in reversed(rewards):
                cumulative = r + gamma * cumulative
                discountedRewards.insert(0, cumulative)

            # Normalize the discounted rewards
            discountedRewards = torch.tensor(discountedRewards, dtype=torch.float32)
            discountedRewards = (discountedRewards - discountedRewards.mean()) / (
                discountedRewards.std() + 1e-8
            )

            averageRewardsBatch += discountedRewards.mean().item()
            averageCriticRewardsBatch += (
                torch.stack(player.values[battle.battle_tag]).mean().item()
            )

            # Calculate the loss using actor-critic
            for log_prob, G, V in zip(
                player.log_probs[battle.battle_tag],
                discountedRewards,
                player.values[battle.battle_tag],
            ):
                advantage = G - V.item()
                actorLoss += -log_prob * advantage

            for G, V in zip(discountedRewards, player.values[battle.battle_tag]):
                criticLoss += (V - G).pow(2)

            # Normalize the critic loss by the episode
            criticLoss /= len(discountedRewards)

        # Normalize the actor loss by the number of battles
        actorLoss /= len(player.battles)

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
        print(f"{batch+1:0{len(str(nBatches))}d}/{nBatches}", end=" ")
        print(
            f"Player {player.username} won {percentage*100:.2f}% of battles.", end=" "
        )
        print(
            f"Actor Loss: {actorLoss.item():>10.4f}, Critic Loss: {criticLoss.item():>10.4f}",
            end=" ",
        )
        print(
            f"Average Rewards: {averageRewardsBatch / len(player.battles):>9.4f}, "
            f"Average Critic Rewards: {averageCriticRewardsBatch / len(player.battles):>9.4f}"
        )

        victoryPercentage.append(percentage)
        actorLosses.append(actorLoss.item())
        criticLosses.append(criticLoss.item())
        averageRewards.append(averageRewardsBatch / len(player.battles))
        averageCriticRewards.append(averageCriticRewardsBatch / len(player.battles))

    # Plot the victory percentage
    plt.plot(victoryPercentage)
    plt.xlabel("Batches")
    plt.ylabel("Victory Percentage")
    plt.title("Victory Percentage Over Batches")

    # Plot the losses
    plt.figure()
    plt.plot(actorLosses)
    plt.xlabel("Batches")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss Over Batches")

    plt.figure()
    plt.plot(criticLosses)
    plt.xlabel("Batches")
    plt.ylabel("Critic Loss")
    plt.title("Critic Loss Over Batches")

    # Plot the average rewards
    plt.figure()
    plt.plot(averageRewards, label="Average Rewards")
    plt.plot(averageCriticRewards, label="Average Critic Rewards")
    plt.xlabel("Batches")
    plt.ylabel("Average Rewards")
    plt.title("Average Rewards Over Batches")
    plt.legend()

    plt.show()


if __name__ == "__main__":

    if os.name == "nt":
        # For Windows
        os.system("cls")
    else:
        # For Linux/macOS
        os.system("clear")

    import ignorePokeEnvProblems

    asyncio.run(main())
