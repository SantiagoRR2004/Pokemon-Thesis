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


async def main():
    serverControl.startServer()

    with open("randomTeam1.txt", "r") as f:
        random_team1 = f.read()

    with open("randomTeam2.txt", "r") as f:
        random_team2 = f.read()

    model = NeuralNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # We create a random player
    player = AIPlayer(
        battle_format="gen9anythinggoes", team=random_team1, network=model
    )

    # We create another random player
    second_player = RandomPlayer(battle_format="gen9anythinggoes", team=random_team2)

    victoryPercentage = []

    for _ in range(10):

        # Reset the player for new epochs
        player.reset()

        # Reset the battle counters
        player.reset_battles()
        second_player.reset_battles()

        # Run n_battles (epochs)
        await player.battle_against(second_player, n_battles=32)

        loss = 0
        gamma = 0.99  # discount factor (the far future is very important)
        initialEpisode = 0

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

            # Calculate the loss
            for log_prob, G in zip(
                player.log_probs[initialEpisode:nEpisodes], discountedRewards
            ):
                loss += -log_prob * G

            initialEpisode = nEpisodes

        # Normalize the loss
        loss /= len(player.battles)

        # Backpropagation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        percentage = player.n_won_battles / player.n_finished_battles

        # We can now print the results of the battles
        print(f"Player {player.username} won {percentage*100:.2f}% of battles")

        victoryPercentage.append(percentage)

    # Plot the victory percentage
    plt.plot(victoryPercentage)
    plt.xlabel("Batches")
    plt.ylabel("Victory Percentage")
    plt.title("Victory Percentage Over Batches")
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
