from MyAIPlayer import AIPlayer
import serverControl
import asyncio
import neat
from MyRandomPlayer import CustomRandomPlayer


async def playGame(
    neuralNetwork1: neat.nn.FeedForwardNetwork,
    neuraNetwork2: neat.nn.FeedForwardNetwork,
) -> int:
    """
    This is the function that checks if a neural network is better than another one

    Args:
        neuralNetwork1 (neat.nn.FeedForwardNetwork): The first neural network
        neuraNetwork2 (neat.nn.FeedForwardNetwork): The second neural network

    Returns:
        int: 1 if the first neural network is better, 0 otherwise
    """
    player1 = AIPlayer(battle_format="gen9randombattle", network=neuralNetwork1)
    player2 = AIPlayer(network=neuraNetwork2)

    await player1.battle_against(player2, n_battles=1)

    if player1.n_won_battles > player2.n_won_battles:
        return 1
    else:
        return 0


async def main():
    serverControl.startServer()

    # We create a random player
    player = CustomRandomPlayer(
        battle_format="gen9randombattle",
    )

    second_player = CustomRandomPlayer()

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


if __name__ == "__main__":
    asyncio.run(main())
