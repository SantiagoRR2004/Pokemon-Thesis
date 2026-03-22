from poke_env.ps_client.server_configuration import ServerConfiguration
from otherPlayers.maxDamagePlayer import MaxRandomDamagePlayer
from players.AbstractAIPlayer import AbstractAIPlayer
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
import pandas as pd
import pokemons
import players
import actors
import torch
import moves
import os

createdPlayers = set()


def getUnusedName(name: str) -> str:
    """
    Returns an unused name based on the given name.

    Need to use n as separator because showdown says:
        - Experiment2_2 == Experiment22
        - Experiment2@2 == Experiment22

    Args:
        - name (str): The base name to use for generating an unused name.

    Returns:
        - str: An unused name based on the given name.
    """
    if name not in createdPlayers:
        createdPlayers.add(name)
        return name

    number = 2

    while f"{name}n{number}" in createdPlayers:
        number += 1

    createdPlayers.add(f"{name}n{number}")

    return f"{name}n{number}"


def getAnyPlayer(choice: str, **args) -> AbstractAIPlayer:
    """
    Returns a player instance based on the given choice.

    Args:
        - choice (str): The type of player to return. Can be "random", "maxDamage", or "experiment{n}".
        - args: Additional positional arguments to pass to the player constructor.

    Returns:
        - AbstractAIPlayer: An instance of AbstractAIPlayer corresponding to the choice.
    """
    if choice == "random":
        return getRandomPlayer(**args)
    elif choice == "maxDamage":
        return getRandomMaxDamagePlayer(**args)
    elif choice.startswith("experiment"):
        n = int(choice[len("experiment") :])
        return getPlayerExperiment(n, **args)

    raise ValueError(
        f"Invalid choice: {choice}. Expected 'random', 'maxDamage', or an integer."
    )


def getRandomPlayer(*, args: dict = {}, **kwargs) -> RandomPlayer:
    """
    Returns a RandomPlayer instance with the given arguments.

    Args:
        - args (dict): A dictionary of arguments to pass to the RandomPlayer constructor.
        - kwargs: Not used, but included to allow for additional
            arguments without breaking the function.

    Returns:
        - RandomPlayer: An instance of RandomPlayer.
    """
    if args.get("account_configuration"):
        return RandomPlayer(**args)

    newArgs = {
        "account_configuration": AccountConfiguration(getUnusedName("Random"), None)
    } | args
    return RandomPlayer(**newArgs)


def getRandomMaxDamagePlayer(*, args: dict = {}, **kwargs) -> MaxRandomDamagePlayer:
    """
    Returns a MaxDamagePlayer instance with the given arguments.

    Args:
        - args (dict): A dictionary of arguments to pass to the MaxDamagePlayer constructor.
        - kwargs: Not used, but included to allow for additional
            arguments without breaking the function.

    Returns:
        - MaxDamagePlayer: An instance of MaxDamagePlayer.
    """
    if args.get("account_configuration"):
        return MaxRandomDamagePlayer(**args)

    newArgs = {
        "account_configuration": AccountConfiguration(getUnusedName("MaxDamage"), None)
    } | args
    return MaxRandomDamagePlayer(**newArgs)


def getPlayerExperiment(
    n: int,
    serverConfig: ServerConfiguration,
    concurrentBattles: int = os.cpu_count(),
    **kwargs,
) -> AbstractAIPlayer:
    """
    This function returns a player for a given experiment number.

    Args:
        - n (int): The experiment number for which to return the player.
        - serverConfig (ServerConfiguration): The server configuration to use for the player.
        - concurrentBattles (int): The number of concurrent battles the player
            allows (default is the number of CPU cores).
        - kwargs: Not used, but included to allow for additional
            arguments without breaking the function.

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
        account_configuration=AccountConfiguration(
            getUnusedName(f"Experiment{n}"), None
        ),
    )

    actor: actors.AbstractActor = getattr(actors, data["actor"])(player)
    actor.load_state_dict(
        torch.load(
            modelFile,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )
    actor.eval()

    # Set the correct network for the player
    player.neuralNetwork = actor

    return player
