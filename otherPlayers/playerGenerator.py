from poke_env.player import RandomPlayer


def getRandomPlayer(args: dict = {}) -> RandomPlayer:
    """
    Returns a RandomPlayer instance with the given arguments.

    Args:
        - args (dict): A dictionary of arguments to pass to the RandomPlayer constructor.

    Returns:
        - RandomPlayer: An instance of RandomPlayer.
    """
    return RandomPlayer(**args)
