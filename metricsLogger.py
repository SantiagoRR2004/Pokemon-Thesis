import matplotlib.pyplot as plt
import pandas as pd
import os


def saveData(
    *,
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
    df.to_parquet(os.path.join(dataDirectory, "experiment.parquet"), index=False)


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

    # Plot the victory percentage
    plt.figure()
    for name, df in files.items():
        if "victoryPercentage" in df.columns:
            smoothed = (
                df["victoryPercentage"].rolling(window=windowSize, center=True).mean()
            )
            plt.plot(smoothed, label=name)
    plt.xlabel("Epochs")
    plt.ylabel("Victory Percentage")
    sortLegend()
    plt.title("Victory Percentage Over Epochs")

    # Plot the losses
    plt.figure()
    for name, df in files.items():
        if "actorLosses" in df.columns:
            smoothed = df["actorLosses"].rolling(window=windowSize, center=True).mean()
            plt.plot(smoothed, label=f"{name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    sortLegend()
    plt.title("Actor Loss Over Epochs")

    plt.figure()
    for name, df in files.items():
        if "criticLosses" in df.columns:
            smoothed = df["criticLosses"].rolling(window=windowSize, center=True).mean()
            plt.plot(smoothed, label=f"{name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    sortLegend()
    plt.title("Critic Loss Over Epochs")

    # Plot the average rewards
    plt.figure()
    for name, df in files.items():
        if "averageRewards" in df.columns:
            smoothed = (
                df["averageRewards"].rolling(window=windowSize, center=True).mean()
            )
            plt.plot(smoothed, label=f"{name} Average Rewards")
        if "averageCriticRewards" in df.columns:
            smoothed = (
                df["averageCriticRewards"]
                .rolling(window=windowSize, center=True)
                .mean()
            )
            plt.plot(smoothed, label=f"{name} Average Critic Rewards")
    plt.xlabel("Epochs")
    plt.ylabel("Average Rewards")
    sortLegend()
    plt.title("Average Rewards Over Epochs")

    # Plot the number of turns
    plt.figure()
    for name, df in files.items():
        if "nTurns" in df.columns:
            smoothed = df["nTurns"].rolling(window=windowSize, center=True).mean()
            plt.plot(smoothed, label=name)
    plt.xlabel("Epochs")
    plt.ylabel("Number of Turns")
    sortLegend()
    plt.title("Number of Turns Over Epochs")

    plt.show()


if __name__ == "__main__":

    graphAllExperiments(100)
