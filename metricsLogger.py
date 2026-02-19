from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pokemons
import players
import moves
import os
import re


class MetricsLogger:
    """
    A class to log and graph the training metrics of the reinforcement learning agent.

    The metrics include:
        - victoryPercentage: The percentage of victories in the evaluation episodes.
        - actorLosses: The loss of the actor network during training.
        - criticLosses: The loss of the critic network during training.
        - averageRewards: The average rewards obtained in the evaluation episodes.
        - averageCriticRewards: The average rewards obtained by the critic in the evaluation episodes.
        - nTurns: The average number of turns taken in the evaluation episodes.
    """

    # Columns that depend on others
    INVALID_COLUMNS = ["fileName", "TrainingMethod", "nInputs"]

    def __init__(self) -> None:
        """
        Initialize the MetricsLogger by loading the existing data from the data directory.

        Args:
            - None

        Returns:
            - None
        """
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        self.dataDirectory = os.path.join(currentDirectory, "data")
        self.graphDirectory = os.path.join(currentDirectory, "graphs")

        # Ensure the graph directory exists
        os.makedirs(self.graphDirectory, exist_ok=True)

        # List all parquet files in the data directory
        files = {
            fileName[: -len(".parquet")]: pd.read_parquet(
                os.path.join(self.dataDirectory, fileName)
            )
            for fileName in os.listdir(self.dataDirectory)
            if fileName.endswith(".parquet")
        }
        # Sort files by name
        files = dict(sorted(files.items(), key=lambda item: item[0]))
        self.files = files

        # Load the experiments data
        self.experimentData = pd.read_csv(
            os.path.join(self.dataDirectory, "experiments.csv")
        )

        # Ccheck that there are no duplicates
        assert not self.experimentData.duplicated(
            subset=self.experimentData.columns.difference(self.INVALID_COLUMNS)
        ).any(), "There are duplicate rows in the experiments.csv file."

        # Remove rows with fileName that is not in the files dictionary
        self.experimentData = self.experimentData[
            self.experimentData["fileName"].isin(self.files.keys())
        ].copy()

        # Create the comparisons DataFrame
        self.calculateComparisons()

        # Calculate the best parameters
        self.calculateBestParameters()

    @staticmethod
    def relativeQuality(A: pd.DataFrame, B: pd.DataFrame) -> float:
        """
        Calculate the relative quality of two sets of metrics.
        It will be a score between -1 and 1, where 1 means A is better than B,
        -1 means B is better than A, and 0 means they are equal.

        Args:
            - A (pd.DataFrame): The first set of metrics.
            - B (pd.DataFrame): The second set of metrics.

        Returns:
            - float: The relative quality of A compared to B.
        """
        # Find the common number of epochs
        minEpochs = min(len(A["epoch"]), len(B["epoch"]))

        # Reduce each dataframe if necessary
        if len(A["epoch"]) > minEpochs:
            A = A.iloc[:minEpochs].copy()
        if len(B["epoch"]) > minEpochs:
            B = B.iloc[:minEpochs].copy()

        # Calculate the victory percentage score
        weights = np.arange(1, minEpochs + 1) / minEpochs
        diff = A["victoryPercentage"].to_numpy() - B["victoryPercentage"].to_numpy()
        victoryPercentageScore = np.sum(weights * np.sign(diff)) / np.sum(weights)

        return victoryPercentageScore

    def graphHeatmap(self, matrix: pd.DataFrame, fileName: str) -> None:
        """
        This function graphs the matrix as a heatmap.

        Args:
            - matrix (pd.DataFrame): The matrix to graph
            - fileName (str): The name of the file to save the graph to

        Returns:
            - None
        """
        data = matrix.to_numpy()
        labels = matrix.index.astype(str).tolist()

        fig, ax = plt.subplots(figsize=(len(labels), len(labels)))
        im = ax.imshow(data, aspect="equal", cmap="coolwarm")

        # Add numbers inside the cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Ensure the value is not NaN
                if not np.isnan(data[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

        # Remove x-axis labels
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.grid(False)

        plt.title(fileName)
        plt.colorbar(im)

        # Save the figure
        text = fileName.replace(" ", "")
        plt.savefig(
            os.path.join(
                self.graphDirectory, f"Column{text[0].upper() + text[1:]}.png"
            ),
            bbox_inches="tight",
        )
        plt.close()

    def calculateComparisons(self) -> None:
        """
        Calculate the relative quality of all pairs of files and store the results in a DataFrame.

        Args:
            - None

        Returns:
            - None
        """
        comparisonsDF = pd.DataFrame(
            np.nan, index=list(self.files.keys()), columns=self.files.keys()
        )

        # Iterate across all pairs of files
        for i in range(len(comparisonsDF.columns)):
            for j in range(i + 1, len(comparisonsDF.columns)):

                data1 = self.files[comparisonsDF.columns[i]]
                data2 = self.files[comparisonsDF.columns[j]]

                # Calculate score
                score = self.relativeQuality(data1, data2)

                # Skew-symmetric matrix
                comparisonsDF.iloc[i, j] = score
                comparisonsDF.iloc[j, i] = -score

        self.comparisonsDF = comparisonsDF

    def calculateBestParameters(self) -> None:
        """
        Calculate the best parameters for each variable in the experiments.csv file

        Args:
            - None

        Returns:
            - None
        """
        # Only use non-random experiments
        nonRandomExperiments = self.experimentData[
            ~self.experimentData["fileName"].str.contains("random")
        ].copy()

        self.bestParameters = {}

        for column in nonRandomExperiments.columns.difference(self.INVALID_COLUMNS):

            parameterValues = nonRandomExperiments[column].unique()
            parametersDF = pd.DataFrame(
                [[[] for _ in parameterValues] for _ in parameterValues],
                index=parameterValues,
                columns=parameterValues,
            )

            # Group rows that are identical except for `column` and the invalid columns
            grouped = nonRandomExperiments.groupby(
                list(
                    nonRandomExperiments.columns.difference(
                        [column] + self.INVALID_COLUMNS
                    )
                )
            )

            for _, group in grouped:
                # Need at least 2 rows to compare
                if len(group) >= 2:
                    # Iterate pairwise inside the group
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            row1 = group.iloc[i]
                            row2 = group.iloc[j]

                            value = self.comparisonsDF.loc[
                                row1["fileName"], row2["fileName"]
                            ]

                            parametersDF.loc[row1[column], row2[column]].append(value)
                            parametersDF.loc[row2[column], row1[column]].append(-value)

            # Remove rows and columns with all empty lists
            parametersDF = parametersDF[
                parametersDF.map(lambda x: len(x) > 0).any(axis=1)
            ]
            parametersDF = parametersDF.loc[
                :, (parametersDF.map(lambda x: len(x) > 0)).any(axis=0)
            ]

            # If there are any values
            if not parametersDF.empty:
                # Average the values in each cell or np.nan
                parametersDF = parametersDF.map(
                    lambda x: np.mean(x) if len(x) > 0 else np.nan
                )

                # Sort rows in ascending order
                allNumeric = all(
                    (x in ["inf", "-inf"])
                    or not np.isnan(pd.to_numeric(x, errors="coerce"))
                    for x in parameterValues
                )

                parametersDF = parametersDF.sort_index(
                    key=lambda idx: [float(x) if allNumeric else x for x in idx]
                )

                # Graph the parameters
                self.graphHeatmap(parametersDF, column)

                # Least-squares score method
                n = parametersDF.shape[0]
                A = np.eye(n) - np.ones((n, n)) / n
                b = parametersDF.sum(axis=1).values
                x_ls = np.linalg.lstsq(A, b, rcond=None)[0]

                ranking = pd.Series(x_ls, index=parametersDF.index).sort_values(
                    ascending=False
                )

            else:
                # If there are no values, we just give a uniform ranking
                ranking = pd.Series(1, index=parameterValues)

            # Add missing values as 1 to make sure they appear more
            ranking = ranking.reindex(parameterValues, fill_value=1)

            # Apply softmax to get probabilities
            ranking = np.exp(ranking) / np.sum(np.exp(ranking))

            self.bestParameters[column] = ranking

    def graphAllExperiments(self) -> None:
        """
        Graph the multiple metrics for the experiments.

        Args:
            - None

        Returns:
            - None
        """
        # Window size for smoothing
        windowSize = 100

        # Get the important files
        top10 = (
            self.comparisonsDF.sum(axis=1).sort_values(ascending=False).head(10).index
        )
        worst = (
            self.comparisonsDF.sum(axis=1).sort_values(ascending=False).tail(1).index
        )
        files = {f: self.files[f] for f in top10.union(worst)}

        # Plot the victory percentage
        fig = go.Figure()
        for name, df in files.items():
            if "victoryPercentage" in df.columns:
                smoothed = (
                    df["victoryPercentage"]
                    .rolling(window=windowSize, center=True)
                    .mean()
                )
                fig.add_trace(
                    go.Scatter(y=smoothed, mode="lines", name=name, hoverinfo="name+y")
                )
        fig.update_layout(
            title="Victory Percentage Over Epochs",
            xaxis_title="Epochs",
            yaxis_title="Victory Percentage",
            hovermode="closest",
        )
        fig.show()

        # Plot the losses
        fig = go.Figure()
        for name, df in files.items():
            if "actorLosses" in df.columns:
                smoothed = (
                    df["actorLosses"].rolling(window=windowSize, center=True).mean()
                )
                # Ensure losses are lower
                if (
                    not smoothed.dropna().empty
                    and smoothed.dropna().iloc[-1] < smoothed.dropna().iloc[0]
                ):
                    fig.add_trace(
                        go.Scatter(
                            y=smoothed, mode="lines", name=name, hoverinfo="name+y"
                        )
                    )
        fig.update_layout(
            title="Actor Loss Over Epochs",
            xaxis_title="Epochs",
            yaxis_title="Loss",
            hovermode="closest",
        )
        fig.show()

        fig = go.Figure()
        for name, df in files.items():
            if "criticLosses" in df.columns:
                smoothed = (
                    df["criticLosses"].rolling(window=windowSize, center=True).mean()
                )
                # Ensure losses are lower
                if (
                    not smoothed.dropna().empty
                    and smoothed.dropna().iloc[-1] < smoothed.dropna().iloc[0]
                ):
                    fig.add_trace(
                        go.Scatter(
                            y=smoothed, mode="lines", name=name, hoverinfo="name+y"
                        )
                    )
        fig.update_layout(
            title="Critic Loss Over Epochs",
            xaxis_title="Epochs",
            yaxis_title="Loss",
            hovermode="closest",
        )
        fig.show()

        # Plot the average rewards
        fig = go.Figure()
        for name, df in files.items():
            if "averageRewards" in df.columns:
                smoothed = (
                    df["averageRewards"].rolling(window=windowSize, center=True).mean()
                )
                # Ensure rewards are higher
                if (
                    not smoothed.dropna().empty
                    and smoothed.dropna().iloc[-1] > smoothed.dropna().iloc[0]
                ):
                    fig.add_trace(
                        go.Scatter(
                            y=smoothed, mode="lines", name=f"{name} Average Rewards"
                        )
                    )
                    if "averageCriticRewards" in df.columns:
                        smoothed = (
                            df["averageCriticRewards"]
                            .rolling(window=windowSize, center=True)
                            .mean()
                        )
                        # Ensure rewards are higher
                        if (
                            not smoothed.dropna().empty
                            and smoothed.dropna().iloc[-1] > smoothed.dropna().iloc[0]
                        ):
                            fig.add_trace(
                                go.Scatter(
                                    y=smoothed,
                                    mode="lines",
                                    name=f"{name} Average Critic Rewards",
                                )
                            )
        fig.update_layout(
            title="Average Rewards Over Epochs",
            xaxis_title="Epochs",
            yaxis_title="Average Rewards",
            hovermode="closest",
        )
        fig.show()

        # Plot the number of turns
        fig = go.Figure()
        for name, df in files.items():
            if "nTurns" in df.columns:
                smoothed = df["nTurns"].rolling(window=windowSize, center=True).mean()
                fig.add_trace(
                    go.Scatter(y=smoothed, mode="lines", name=name, hoverinfo="name+y")
                )
        fig.update_layout(
            title="Number of Turns Over Epochs",
            xaxis_title="Epochs",
            yaxis_title="Number of Turns",
            hovermode="closest",
        )
        fig.show()

    def graphVictoryPercentage(self) -> None:
        """
        Graph the victory percentage for the different
        variables in the experiments.csv file.

        Args:
            - None

        Returns:
            - None
        """
        for column in self.experimentData.columns:

            if column != "fileName":

                fig = go.Figure()
                fig.update_layout(
                    title=f"Victory Percentage for {column}",
                    xaxis_title="Epochs",
                    yaxis_title="Victory Percentage",
                    hovermode="closest",
                )

                # Group values by the column
                grouped = self.experimentData.groupby(column)

                for name, group in grouped:

                    smoothedTotal = []

                    # Now we average the data of all files in the group
                    for fileName in group["fileName"]:

                        if self.files.get(fileName) is not None:
                            df = self.files[fileName]
                            smoothed = (
                                df["victoryPercentage"]
                                .rolling(window=100, center=True)
                                .mean()
                            )
                            smoothedTotal.append(smoothed)

                    if smoothedTotal:
                        combined = pd.concat(smoothedTotal, axis=1).mean(axis=1)

                        fig.add_trace(go.Scatter(y=combined, mode="lines", name=name))

                fig.show()

    def obtainNewExperiment(self) -> dict:
        """
        Obtain a new experiment by sampling from the best parameters.

        Args:
            - None

        Returns:
            - dict: A dictionary with the sampled parameters
        """
        return {
            column: series.sample(n=1, weights=series).index[0]
            for column, series in self.bestParameters.items()
        }

    def createNewExperiments(self, n: int = 5) -> list:
        """
        Create n new experiments by sampling from the best parameters.

        Args:
            - n (int): The number of new experiments to create

        Returns:
            - list: A list of dictionaries with the sampled parameters for each experiment
        """
        # Obtain all experiments
        allExperiments = pd.read_csv(
            os.path.join(self.dataDirectory, "experiments.csv")
        )
        latestExperimentInt = max(
            [
                int(f.split("experiment")[-1].split(".")[0])
                for f in allExperiments["fileName"]
                if "experiment" in f
            ]
        )

        nAdded = 0

        while nAdded < n:
            # Sample new parameters
            params = self.obtainNewExperiment()

            # Check that the experiment does not already exist
            exists = (
                (allExperiments[list(params)] == pd.Series(params)).all(axis=1)
            ).any()

            if not exists:
                # Create a new file name
                fileName = f"experiment{latestExperimentInt + nAdded + 1}"
                params["fileName"] = fileName

                # The TrainingMethod
                if pd.isna(params["critic"]):
                    params["TrainingMethod"] = "actor"
                else:
                    params["TrainingMethod"] = "actorCritic"

                # Add nInputs as 0
                params["nInputs"] = 0

                # Add to allExperiments
                allExperiments = pd.concat(
                    [allExperiments, pd.DataFrame([params])], ignore_index=True
                )

                nAdded += 1

        # Correctly calculate the nInputs column
        for index, row in allExperiments.iterrows():

            try:
                # Least printing possible
                with open(os.devnull, "w") as fnull:
                    with redirect_stdout(fnull), redirect_stderr(fnull):
                        move = getattr(moves, row.move)()
                        pokemon = getattr(pokemons, row.pokemon)(move)
                        player = getattr(players, row.player)(
                            network="BlaBlaBla", pokemonFeatureExtractor=pokemon
                        )

                allExperiments.at[index, "nInputs"] = player.getNumberOfInputs()

            except AttributeError as e:
                continue

        def naturalKey(s):
            return [
                int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)
            ]

        # Order the rows by fileName
        allExperiments = allExperiments.sort_values(
            by="fileName", key=lambda x: x.map(naturalKey)
        ).reset_index(drop=True)

        # Save the new experiments
        allExperiments.to_csv(
            os.path.join(self.dataDirectory, "experiments.csv"), index=False
        )


def saveData(
    *,
    fileName: str = "experiment",
    victoryPercentage: list,
    actorLosses: list = None,
    criticLosses: list = None,
    averageRewards: list = None,
    averageCriticRewards: list = None,
    nTurns: list = None,
) -> None:
    """
    Save the training metrics to a parquet file.

    The args should be lists containing the metrics for each epoch.

    Returns:
        - None
    """
    dataDict = {
        "epoch": list(range(len(victoryPercentage))),
        "victoryPercentage": victoryPercentage,
        "actorLosses": actorLosses,
        "criticLosses": criticLosses,
        "averageRewards": averageRewards,
        "averageCriticRewards": averageCriticRewards,
        "nTurns": nTurns,
    }

    df = pd.DataFrame()

    for key, value in dataDict.items():
        if value is not None:
            df[key] = value

    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataDirectory = os.path.join(currentDirectory, "data")
    # Ensure the directory exists
    os.makedirs(dataDirectory, exist_ok=True)

    # Save as a parquet file
    df.to_parquet(os.path.join(dataDirectory, fileName + ".parquet"), index=False)


def sortLegend() -> None:
    """
    Sort the legend of the current plot by label name.

    Args:
        - None

    Returns:
        - None
    """
    # Get current legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Sort labels and handles by label name
    sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)

    # Apply sorted legend
    plt.legend(sorted_handles, sorted_labels)


if __name__ == "__main__":
    logger = MetricsLogger()
    logger.graphAllExperiments()
    logger.graphVictoryPercentage()
