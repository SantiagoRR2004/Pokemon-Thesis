import pandas as pd
import os

currentDirectory = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(os.path.join(currentDirectory, "experiments.csv"))

subsetCols = [col for col in data.columns if col != "fileName"]


# Check that all rows are unique
if data.duplicated(subset=subsetCols).any():
    print("Duplicate rows found:")
    print(data[data.duplicated(subset=subsetCols)])
else:
    print("No duplicate rows found.")
