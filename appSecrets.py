import json


def getShowdownUsername() -> str:
    """
    Gets the username from the secrets.json file

    Args:
        - None

    Returns:
        - str: The username to be used for Showdown
    """
    with open("secrets.json") as f:
        data = json.load(f)
        return data["username"]


def getShowdownPassword() -> str:
    """
    Gets the password from the secrets.json file

    Args:
        - None

    Returns:
        - str: The password to be used for Showdown
    """
    with open("secrets.json") as f:
        data = json.load(f)
        return data["password"]
