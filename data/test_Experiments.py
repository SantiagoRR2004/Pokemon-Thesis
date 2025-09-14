from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
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


# Find and display the groups
groups = findSingleDifferenceGroups(data, subsetCols)

# Summary
totalGroups = sum(len(groups) for groups in groups.values())
print(f"Found {totalGroups} groups across {len(groups)} columns")

for varyingColumn, groups in groups.items():
    print(f"  {varyingColumn}: {len(groups)} groups")


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
