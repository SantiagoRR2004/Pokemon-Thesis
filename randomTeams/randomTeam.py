import os
import random


def selectTeam(team: int) -> str:
    """
    Selects the team based on the provided team number.

    Args:
        - team (int): The team number to select.

    Returns:
        - str: The content of the team file.
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(currentDirectory, f"randomTeam{team}.txt")

    with open(file, "r") as f:
        team_data = f.read()

    return team_data


def selectRandomTeam(range: int) -> str:
    """
    Selects a random team from the available teams.

    Args:
        - range (int): The number of available teams.

    Returns:
        - str: The content of the randomly selected team file.
    """
    team = random.randint(1, range)
    return selectTeam(team)
