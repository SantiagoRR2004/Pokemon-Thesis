from contextlib import redirect_stdout, redirect_stderr
from typing import Tuple
import pandas as pd
import numpy as np
import pokemons
import players
import moves
import os
import re

currentDirectory = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(os.path.join(currentDirectory, "experiments.csv"))

specialColumns = ["fileName", "nInputs", "TrainingMethod"]

subsetCols = [col for col in data.columns if col not in specialColumns]


# Check that all rows are unique
if data.duplicated(subset=subsetCols).any():
    print("Duplicate rows found:")
    print(data[data.duplicated(subset=subsetCols)])
else:
    print("No duplicate rows found.")


def findSingleDifferenceGroups(df: pd.DataFrame, columnsToCompare: list) -> dict:
    """
    Find groups of rows that have the same values in all columns except one.

    Args:
        - df (pd.DataFrame): The DataFrame to analyze.
        - columnsToCompare (list): List of columns to consider for comparison.

    Returns:
        - dict: A dictionary where keys are the varying columns
            and values are lists of rows
    """
    groupsByVaryingColumn = {}

    for varyingCol in columnsToCompare:
        # Get all other columns (fixed columns)
        fixedCols = [col for col in columnsToCompare if col != varyingCol]

        # Group by fixed columns
        grouped = df.groupby(fixedCols)

        # Find groups with multiple rows
        multiRowGroups = []
        for _, groupDf in grouped:
            if len(groupDf) > 1:
                # Check if they actually differ in the varying column
                uniqueValues = groupDf[varyingCol].nunique()
                if uniqueValues > 1:
                    multiRowGroups.append(groupDf.to_dict("records"))

        if multiRowGroups:
            groupsByVaryingColumn[varyingCol] = multiRowGroups

    return groupsByVaryingColumn


def makeVictoryMatrix(groupData: dict) -> Tuple[np.ndarray, list]:
    """
    Create a matrix of victory percentages for experiments.

    Args:
        - groupData (dict): A dictionary where keys are experiment names
            and values are DataFrames with experiment results.

    Returns:
        - Tuple[np.ndarray, list]: A tuple containing the victory matrix
            and the list of experiment names (keys).
    """
    victories = {}

    for name, df in groupData.items():
        if "victoryPercentage" in df.columns:
            victories[name] = (
                df["victoryPercentage"][:1000]
                .rolling(window=100, center=True)
                .mean()
                .dropna()
                .tolist()
            )

    nWindows = 1000 - 100 + 1

    keys = list(groupData.keys())

    arr = np.full((len(victories), nWindows), -np.inf)  # pad with -inf
    for i, k in enumerate(keys):
        arr[i, :] = victories[k]

    return arr, keys


def rankExperiments(groupData: dict) -> list:
    """
    Rank experiments based on their victory percentage.

    Args:
        - groupData (dict): A dictionary where keys are experiment names
            and values are DataFrames with experiment results.

    Returns:
        - list: A list of the ordered keys of the input dictionary,
            first being the best experiment.
    """
    arr, keys = makeVictoryMatrix(groupData)

    ranking = []
    remaining = np.arange(len(keys))

    while arr.shape[0] > 0:
        # Column-wise maximum
        colMax = np.nanmax(arr, axis=0)

        # Count how many times each row equals the column max
        counts = np.sum(arr == colMax, axis=1)

        # Pick row with maximum count
        bestIdx = np.argmax(counts)
        ranking.append(keys[remaining[bestIdx]])

        # Remove that row
        arr = np.delete(arr, bestIdx, axis=0)
        remaining = np.delete(remaining, bestIdx)

    return ranking


# Find the groups
groups = findSingleDifferenceGroups(data, subsetCols)

# Delete all the rows that have random in the fileName column
groups = {
    k: v
    for k, v in groups.items()
    if all("random" not in row["fileName"] for group in v for row in group)
}

for differentColumn, groups in groups.items():
    for g in groups:

        groupData = {
            row[differentColumn]: pd.read_parquet(
                os.path.join(currentDirectory, row["fileName"] + ".parquet")
            )
            for row in g
            if row["fileName"] + ".parquet" in os.listdir(currentDirectory)
        }

        if len(groupData) > 1:
            # rankExperiments(groupData)
            pass
            # I can use this later

presetValues = {"nTeams": "inf", "nEpisodes": "64", "gamma": "0.99"}
# Get the rows with those values

rows = data[
    (data["nTeams"] == presetValues["nTeams"])
    & (data["nEpisodes"] == presetValues["nEpisodes"])
    & (data["gamma"] == presetValues["gamma"])
]

rowData = {
    row["fileName"]: pd.read_parquet(
        os.path.join(currentDirectory, row["fileName"] + ".parquet")
    )
    for _, row in rows.iterrows()
    if row["fileName"] + ".parquet" in os.listdir(currentDirectory)
}

arr, keys = makeVictoryMatrix(rowData)

colMax = np.nanmax(arr, axis=0)
maxCounts = np.sum(arr == colMax, axis=1)
probabilities = maxCounts / maxCounts.sum()

# Get latest experiment int for experiment{int} in filename
latestExperimentInt = max(
    [
        int(f.split("experiment")[-1].split(".")[0])
        for f in data["fileName"]
        if "experiment" in f
    ]
)

nAdded = 0

while nAdded < 5:
    chosenIdx = np.random.choice(np.arange(len(keys)), p=probabilities)
    chosenKey = keys[chosenIdx]

    newRow = rows[rows["fileName"] == chosenKey].copy()

    # Randomly change one of the columns to a different value
    colToChange = np.random.choice(subsetCols)
    possibleValues = data[colToChange].unique()
    newValue = np.random.choice(
        [v for v in possibleValues if v != newRow.iloc[0][colToChange]]
    )
    if isinstance(newValue, (str, np.str_)) and newValue.lower() == "nan":
        newValue = ""
    newRow.loc[newRow.index[0], colToChange] = newValue

    # Change the fileName to experiment{latestExperimentInt + nAdded + 1}
    newRow.loc[newRow.index[0], "fileName"] = (
        f"experiment{latestExperimentInt + nAdded + 1}"
    )

    if newValue not in ["random", 0, "r1", "r2", "r10", "r64", "r100", "rinf"]:

        newData = pd.concat([data, newRow], ignore_index=True)

        newData = newData.drop_duplicates(subset=subsetCols, keep="first")

        new_row_index = len(newData) - 1  # Index of the newly added row
        is_duplicate = newData.duplicated(subset=subsetCols, keep="first").iloc[
            new_row_index
        ]
        is_duplicate2 = any(
            (data[subsetCols] == newRow[subsetCols].iloc[0]).all(axis=1)
        )

        if len(newData) == len(data) or is_duplicate or is_duplicate2:
            # If the new row is a duplicate, don't add it
            continue
        else:
            # Add the modified row to the dataframe
            data = newData
            nAdded += 1


# Correctly calculate the nInputs column
for index, row in data.iterrows():

    try:

        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):

                move = getattr(moves, row.move)()
                pokemon = getattr(pokemons, row.pokemon)(move)
                player = getattr(players, row.player)(
                    network="BlaBlaBla", pokemonFeatureExtractor=pokemon
                )

        data.at[index, "nInputs"] = player.getNumberOfInputs()

    except AttributeError as e:
        continue

    if pd.isna(row["critic"]):
        data.at[index, "TrainingMethod"] = "actor"
    else:
        data.at[index, "TrainingMethod"] = "actorCritic"


def naturalKey(s):
    return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)]


# Order the rows by fileName
data = data.sort_values(by="fileName", key=lambda x: x.map(naturalKey)).reset_index(
    drop=True
)

data.to_csv(os.path.join(currentDirectory, "experiments.csv"), index=False)
