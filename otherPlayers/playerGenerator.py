from poke_env.ps_client.server_configuration import ServerConfiguration
from otherPlayers.maxDamagePlayer import MaxRandomDamagePlayer
from players.AbstractAIPlayer import AbstractAIPlayer
from poke_env.player import RandomPlayer
import pandas as pd
import pokemons
import players
import actors
import torch
import moves
import os


def getRandomPlayer(args: dict = {}) -> RandomPlayer:
    """
    Returns a RandomPlayer instance with the given arguments.

    Args:
        - args (dict): A dictionary of arguments to pass to the RandomPlayer constructor.

    Returns:
        - RandomPlayer: An instance of RandomPlayer.
    """
    return RandomPlayer(**args)


def getRandomMaxDamagePlayer(args: dict = {}) -> MaxRandomDamagePlayer:
    """
    Returns a MaxDamagePlayer instance with the given arguments.

    Args:
        - args (dict): A dictionary of arguments to pass to the MaxDamagePlayer constructor.

    Returns:
        - MaxDamagePlayer: An instance of MaxDamagePlayer.
    """
    return MaxRandomDamagePlayer(**args)


def getPlayerExperiment(
    n: int, serverConfig: ServerConfiguration, concurrentBattles: int = 100
) -> AbstractAIPlayer:
    """
    This function returns a player for a given experiment number.

    Args:
        - n (int): The experiment number for which to return the player.
        - serverConfig (ServerConfiguration): The server configuration to use for the player.
        - concurrentBattles (int): The number of concurrent battles the player
            allows (default is 100).

    Returns:
        - AbstractAIPlayer: An instance of AbstractAIPlayer corresponding to the experiment number.
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataPath = os.path.join(os.path.dirname(currentDirectory), "data")
    modelFile = os.path.join(dataPath, "experiments", f"experiment{n}Actor.pth")
    dataFile = os.path.join(dataPath, f"experiments.csv")

    # Check if the model file exists
    if not os.path.exists(modelFile):
        raise ValueError(f"Model file {modelFile} does not exist.")

    allData = pd.read_csv(dataFile)

    # Check if the experiment number exists in the data
    if f"experiment{n}" not in allData["fileName"].values:
        raise ValueError(f"Experiment number {n} does not exist in the data.")

    data = allData[allData["fileName"] == f"experiment{n}"].iloc[0]

    player: AbstractAIPlayer = getattr(players, data["player"])(
        network="BlaBlaBla",
        battle_format="gen9randombattle",
        pokemonFeatureExtractor=getattr(pokemons, data["pokemon"])(
            getattr(moves, data["move"])
        ),
        server_configuration=serverConfig,
        max_concurrent_battles=concurrentBattles,
    )

    actor: actors.AbstractActor = getattr(actors, data["actor"])(player)
    actor.load_state_dict(torch.load(modelFile))
    actor.eval()

    # Set the correct network for the player
    player.neuralNetwork = actor

    return player
