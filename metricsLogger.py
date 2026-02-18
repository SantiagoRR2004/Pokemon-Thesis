import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os


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

    def __init__(self) -> None:
        """
        Initialize the MetricsLogger by loading the existing data from the data directory.

        Args:
            - None

        Returns:
            - None
        """
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        dataDirectory = os.path.join(currentDirectory, "data")

        # List all parquet files in the data directory
        files = {
            fileName[: -len(".parquet")]: pd.read_parquet(
                os.path.join(dataDirectory, fileName)
            )
            for fileName in os.listdir(dataDirectory)
            if fileName.endswith(".parquet")
        }
        # Sort files by name
        files = dict(sorted(files.items(), key=lambda item: item[0]))
        self.files = files

        # Load the experiments data
        self.experimentData = pd.read_csv(
            os.path.join(dataDirectory, "experiments.csv")
        )

        # Create the comparisons DataFrame
        self.calculateComparisons()

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


def graphVictoryPercentage() -> None:
    """
    Graph the victory percentage for the different
    variables in the experiments.csv file.

    Args:
        - None

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataDirectory = os.path.join(currentDirectory, "data")

    # List all parquet files in the data directory
    files = {
        fileName[: -len(".parquet")]: pd.read_parquet(
            os.path.join(dataDirectory, fileName)
        )
        for fileName in os.listdir(dataDirectory)
        if fileName.endswith(".parquet")
    }
    files = dict(sorted(files.items(), key=lambda item: item[0]))  # sort by name

    experimentData = pd.read_csv(os.path.join(dataDirectory, "experiments.csv"))

    for column in experimentData.columns:

        if column != "fileName":

            fig = go.Figure()
            fig.update_layout(
                title=f"Victory Percentage for {column}",
                xaxis_title="Epochs",
                yaxis_title="Victory Percentage",
                hovermode="closest",
            )

            # Group values by the column
            grouped = experimentData.groupby(column)

            for name, group in grouped:

                smoothedTotal = []

                # Now we average the data of all files in the group
                for fileName in group["fileName"]:

                    if files.get(fileName) is not None:
                        df = files[fileName]
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


if __name__ == "__main__":
    logger = MetricsLogger()
    logger.graphAllExperiments()
    graphVictoryPercentage()
