from MyAIPlayer import AIPlayer
import torch
import torch.nn as nn
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

    # We create a random player
    player = AIPlayer(
        battle_format="gen9anythinggoes", team=random_team1, network=NeuralNetwork()
    )

    # We create another random player
    second_player = AIPlayer(
        battle_format="gen9anythinggoes", team=random_team2, network=NeuralNetwork()
    )

    # The battle_against method initiates a battle between two players.
    # Here we are using asynchronous programming (await) to start the battle.
    await player.battle_against(second_player, n_battles=1)

    # We can now print the results of the battles
    print(
        f"Player {player.username} won {player.n_won_battles} out of {player.n_finished_battles} played"
    )
    print(
        f"Player {second_player.username} won {second_player.n_won_battles} out of {second_player.n_finished_battles} played"
    )

    input()


if __name__ == "__main__":

    if os.name == "nt":
        # For Windows
        os.system("cls")
    else:
        # For Linux/macOS
        os.system("clear")

    import ignorePokeEnvProblems

    asyncio.run(main())
