import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import os


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


def graphExperiment(fileName: str) -> None:
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataDirectory = os.path.join(currentDirectory, "data")

    df = pd.read_parquet(os.path.join(dataDirectory, fileName))

    # Plot the victory percentage
    if "victoryPercentage" in df.columns:
        plt.figure()
        plt.plot(df["victoryPercentage"])
        plt.xlabel("Epochs")
        plt.ylabel("Victory Percentage")
        plt.title("Victory Percentage Over Epochs")

    # Plot the losses
    if "actorLosses" in df.columns:
        plt.figure()
        plt.plot(df["actorLosses"])
        plt.xlabel("Epochs")
        plt.ylabel("Actor Loss")
        plt.title("Actor Loss Over Epochs")

    if "criticLosses" in df.columns:
        plt.figure()
        plt.plot(df["criticLosses"])
        plt.xlabel("Epochs")
        plt.ylabel("Critic Loss")
        plt.title("Critic Loss Over Epochs")

    # Plot the average rewards
    if "averageRewards" in df.columns or "averageCriticRewards" in df.columns:
        plt.figure()
        if "averageRewards" in df.columns:
            plt.plot(df["averageRewards"], label="Average Rewards")
        if "averageCriticRewards" in df.columns:
            plt.plot(df["averageCriticRewards"], label="Average Critic Rewards")
        plt.xlabel("Epochs")
        plt.ylabel("Average Rewards")
        plt.title("Average Rewards Over Epochs")
        plt.legend()

    # Plot the number of turns
    if "nTurns" in df.columns:
        plt.figure()
        plt.plot(df["nTurns"])
        plt.xlabel("Epochs")
        plt.ylabel("Number of Turns")
        plt.title("Number of Turns Over Epochs")

    plt.show()


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


def graphAllExperiments(windowSize: int = 1) -> None:
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

    # Plot the victory percentage
    fig = go.Figure()
    for name, df in files.items():
        if "victoryPercentage" in df.columns:
            smoothed = (
                df["victoryPercentage"].rolling(window=windowSize, center=True).mean()
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
            smoothed = df["actorLosses"].rolling(window=windowSize, center=True).mean()
            # Ensure losses are lower
            if (
                not smoothed.dropna().empty
                and smoothed.dropna().iloc[-1] < smoothed.dropna().iloc[0]
            ):
                fig.add_trace(
                    go.Scatter(y=smoothed, mode="lines", name=name, hoverinfo="name+y")
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
            smoothed = df["criticLosses"].rolling(window=windowSize, center=True).mean()
            # Ensure losses are lower
            if (
                not smoothed.dropna().empty
                and smoothed.dropna().iloc[-1] < smoothed.dropna().iloc[0]
            ):
                fig.add_trace(
                    go.Scatter(y=smoothed, mode="lines", name=name, hoverinfo="name+y")
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
                    go.Scatter(y=smoothed, mode="lines", name=f"{name} Average Rewards")
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
    graphAllExperiments(100)
    graphVictoryPercentage()
