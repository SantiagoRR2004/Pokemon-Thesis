import json


def getShowdownUsername():
    with open("secrets.json") as f:
        data = json.load(f)
        return data["username"]


def getShowdownPassword():
    with open("secrets.json") as f:
        data = json.load(f)
        return data["password"]
