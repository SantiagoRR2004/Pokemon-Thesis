from poke_env.player import RandomPlayer
from otherPlayers.maxDamagePlayer import MaxRandomDamagePlayer


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
