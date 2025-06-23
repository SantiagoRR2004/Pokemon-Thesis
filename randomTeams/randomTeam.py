import os
import random
import subprocess


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

    return team_data.strip()


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


def generateTeams(number: int) -> None:
    """
    Generates teams by running an external script.

    Args:
        - number (int): The number of teams to generate.

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(currentDirectory, "generate-teams.sh")
    try:
        subprocess.run([file, str(number)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with error: {e}")
