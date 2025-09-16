from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import numpy as np
import pokemons
import players
import moves
import os

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

    keys = list(groupData.keys())

    arr = np.full((len(victories), 1000), -np.inf)  # pad with -inf
    for i, k in enumerate(keys):
        arr[i, : len(victories[k])] = victories[k]

    ranking = []
    remaining = np.arange(len(victories))

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

            print(rankExperiments(groupData))


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

data.to_csv(os.path.join(currentDirectory, "experiments.csv"), index=False)
