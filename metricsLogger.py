from statsmodels.stats.proportion import proportion_confint
from contextlib import redirect_stdout, redirect_stderr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import serverControl
import pandas as pd
import otherPlayers
import numpy as np
import pokemons
import asyncio
import players
import moves
import torch
import json
import tqdm
import io
import os
import re
import gc


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

    def __init__(
        self,
        infiniteBattles: bool = False,
        maxUncertainty: float = 0.01,
        surrogateGraphs: bool = False,
    ) -> None:
        """
        Initialize the MetricsLogger by loading the existing data from the data directory.

        Args:
            - infiniteBattles (bool): Whether to run infinite battles. If False, it will
                calculate the best parameters for the surrogate data.
            - maxUncertainty (float): The maximum uncertainty to consider for the infinite battles.
                This is a percentage.
            - surrogateGraphs (bool): Whether to calculate the best parameters for the surrogate data.

        Returns:
            - None
        """
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        self.dataDirectory = os.path.join(currentDirectory, "data")
        self.experimentsDirectory = os.path.join(self.dataDirectory, "experiments")
        self.graphDirectory = os.path.join(currentDirectory, "graphs")
        self.infiniteBattles = infiniteBattles
        self.maxUncertainty = maxUncertainty
        self.surrogateRowIndex = 0

        # Ensure the graph directory exists
        os.makedirs(self.graphDirectory, exist_ok=True)

        # List all parquet files in the data directory
        files = {
            fileName[: -len(".parquet")]: pd.read_parquet(
                os.path.join(self.experimentsDirectory, fileName)
            )
            for fileName in os.listdir(self.experimentsDirectory)
            if fileName.endswith(".parquet")
        }
        # Sort files by name
        files = dict(sorted(files.items(), key=lambda item: item[0]))
        self.files = files

        # Load the experiments data
        self.experimentData = pd.read_csv(
            os.path.join(self.dataDirectory, "experiments.csv")
        )

        # Check that there are no duplicates
        assert not self.experimentData.duplicated(
            subset=self.experimentData.columns.difference(self.INVALID_COLUMNS)
        ).any(), "There are duplicate rows in the experiments.csv file."

        # Calculate the missing rows
        missingRows = self.experimentData[
            ~self.experimentData["fileName"].isin(self.files.keys())
        ].copy()
        missingRows.to_csv(os.path.join(self.dataDirectory, "missing.csv"), index=False)

        # Remove rows with fileName that is not in the files dictionary
        self.experimentData = self.experimentData[
            self.experimentData["fileName"].isin(self.files.keys())
        ].copy()

        # Create the comparisons DataFrame from the metrics
        self.calculateComparisons()

        # Create another DataFrame by battling
        self.calculateTournament()

        # Calculate the Bradley-Terry model with the tournament results
        self.bradleyTerryDF = self.bradleyTerry(self.tournamentWonDF)

        # Use surrogate modeling
        self.createSurrogateModel()
        self.surrogateData()

        # Calculate the best parameters for the metrics (to create graphs)
        self.bestParametersLogs = self.calculateBestParameters(
            self.comparisonsDF, "Logs"
        )

        # Calculate the best parameters for the battles
        self.bestParameters = self.calculateBestParameters(
            self.tournamentDF, "Battles", True
        )

        if self.infiniteBattles:
            self.infiniteTournament()

        if surrogateGraphs:
            self.surrogateBestParameters()

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

        weights = np.arange(1, minEpochs + 1) / minEpochs

        # Calculate the victory percentage score
        diff = A["victoryPercentage"].to_numpy() - B["victoryPercentage"].to_numpy()
        victoryPercentageScore = np.sum(weights * np.sign(diff)) / np.sum(weights)
        nScores = 1

        # Calculate the actor loss score
        actorLossScore = 0
        if "actorLosses" in A.columns and "actorLosses" in B.columns:
            diff = A["actorLosses"].to_numpy() - B["actorLosses"].to_numpy()
            actorLossScore = -np.sum(weights * np.sign(diff)) / np.sum(weights)
            nScores += 1

        # Calculate the critic loss score
        criticLossScore = 0
        if "criticLosses" in A.columns and "criticLosses" in B.columns:
            diff = A["criticLosses"].to_numpy() - B["criticLosses"].to_numpy()
            criticLossScore = -np.sum(weights * np.sign(diff)) / np.sum(weights)
            nScores += 1

        # Calculate the average rewards score
        averageRewardsScore = 0
        if "averageRewards" in A.columns and "averageRewards" in B.columns:
            diff = A["averageRewards"].to_numpy() - B["averageRewards"].to_numpy()
            averageRewardsScore = np.sum(weights * np.sign(diff)) / np.sum(weights)
            nScores += 1

        # Calculate the average critic rewards score
        averageCriticRewardsScore = 0
        if "averageCriticRewards" in A.columns and "averageCriticRewards" in B.columns:
            diff = (
                A["averageCriticRewards"].to_numpy()
                - B["averageCriticRewards"].to_numpy()
            )
            averageCriticRewardsScore = np.sum(weights * np.sign(diff)) / np.sum(
                weights
            )
            nScores += 1

        return (
            victoryPercentageScore
            + actorLossScore
            + criticLossScore
            + averageRewardsScore
            + averageCriticRewardsScore
        ) / nScores

    def graphHeatmap(self, matrix: pd.DataFrame, fileName: str) -> None:
        """
        This function graphs the matrix as a heatmap.

        Args:
            - matrix (pd.DataFrame): The matrix to graph
            - fileName (str): The name of the file to save the graph to

        Returns:
            - None
        """
        # Sort columns and rows
        matrix = sortMatrix(matrix)

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

        # Save the figure
        text = "".join([word[0].upper() + word[1:] for word in fileName.split()])
        plt.savefig(
            os.path.join(self.graphDirectory, f"{text}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

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

        # Graph the comparisons matrix (Too big)
        # self.graphHeatmap(comparisonsDF, "Logs All")

    def resetServer(self) -> None:
        """
        Reset the server by stopping and starting it again.

        Args:
            - None

        Returns:
            - None
        """
        if not hasattr(self, "players"):
            self.players = []

        for player in self.players:
            del player

        self.players = []
        gc.collect()
        torch.cuda.empty_cache()

        serverControl.endProcess(self.p)
        self.p.wait()
        self.p = serverControl.startServer()
        self.serverCount = 0

    async def playBattlesAsync(
        self, name1: str, name2: str, nBattles: int = 100
    ) -> float:
        """
        Play battles between two players asynchronously and return the win percentage.

        Args:
            - name1 (str): The name of the first player.
            - name2 (str): The name of the second player.
            - nBattles (int): The number of battles to play.

        Returns:
            - float: The win percentage of the first player.
        """
        arguments = {
            "serverConfig": serverControl.getServerConfiguration(),
            "args": {
                "max_concurrent_battles": os.cpu_count(),
                "server_configuration": serverControl.getServerConfiguration(),
            },
        }

        player1 = otherPlayers.getAnyPlayer(name1, **arguments)
        self.players.append(player1)
        player2 = otherPlayers.getAnyPlayer(name2, **arguments)
        self.players.append(player2)

        # Battle the two players
        await player1.battle_against(player2, n_battles=nBattles)

        return player1.win_rate

    def playBattles(self, name1: str, name2: str, nBattles: int = 100) -> None:
        """
        Play battles between two players and update the tournament results.

        It will reset the server if a timeout occurs or if the number of battles
        played since the last reset exceeds 1000.

        Args:
            - name1 (str): The name of the first player.
            - name2 (str): The name of the second player.
            - nBattles (int): The number of battles to play.

        Returns:
            - None
        """
        if not getattr(self, "p", None):
            # Start the server if it is not already started
            self.p = serverControl.startServer()
            self.serverCount = 0

        if not hasattr(self, "players"):
            self.players = []

        continuePlaying = True

        while continuePlaying:
            try:
                winPercentage = asyncio.run(
                    asyncio.wait_for(
                        self.playBattlesAsync(name1, name2, nBattles), timeout=200
                    )
                )
                continuePlaying = False
                self.serverCount += nBattles

            except asyncio.TimeoutError:
                print(
                    f"Error playing battles between {name1} and {name2}: Timeout",
                    flush=True,
                )
                self.resetServer()

            finally:
                if self.serverCount >= 1000:
                    self.resetServer()

        # Calculate won battles
        won = int(winPercentage * nBattles)

        # Store won battles and total battles
        if pd.isna(self.tournamentWonDF.at[name1, name2]):
            self.tournamentWonDF.at[name1, name2] = won
            self.tournamentPlayedDF.at[name1, name2] = nBattles
        else:
            self.tournamentWonDF.at[name1, name2] += won
            self.tournamentPlayedDF.at[name1, name2] += nBattles

        # The battles are symmetric, so we can fill the other cell as well
        self.tournamentWonDF.at[name2, name1] = (
            self.tournamentPlayedDF.at[name1, name2]
            - self.tournamentWonDF.at[name1, name2]
        )
        self.tournamentPlayedDF.at[name2, name1] = self.tournamentPlayedDF.at[
            name1, name2
        ]

        # Save the tournament results to avoid losing data
        sortMatrix(self.tournamentWonDF).to_csv(self.battlesWonFile, index=True)
        sortMatrix(self.tournamentPlayedDF).to_csv(self.battlesPlayedFile, index=True)

    def calculateTournament(self) -> None:
        """
        Battle all the players against each other and store
        the results in two separate DataFrames (won and played battles).

        Args:
            - None

        Returns:
            - None
        """
        self.battlesWonFile = os.path.join(self.dataDirectory, "battlesWon.csv")
        self.battlesPlayedFile = os.path.join(self.dataDirectory, "battlesPlayed.csv")

        files = [
            fileName[: -len("Actor.pth")]
            for fileName in os.listdir(self.experimentsDirectory)
            if fileName.endswith("Actor.pth")
        ] + ["random", "maxDamage"]

        if os.path.exists(self.battlesWonFile) and os.path.exists(
            self.battlesPlayedFile
        ):
            tournamentWonDF = pd.read_csv(self.battlesWonFile, index_col=0).astype(
                "Int64"
            )
            tournamentPlayedDF = pd.read_csv(
                self.battlesPlayedFile, index_col=0
            ).astype("Int64")

            # Assert that the index and columns have the same elements
            assert set(tournamentWonDF.index) == set(
                tournamentWonDF.columns
            ), "Square matrix required."
            assert set(tournamentPlayedDF.index) == set(
                tournamentPlayedDF.columns
            ), "Square matrix required."

        else:
            # Create DataFrames to store the results
            tournamentWonDF = pd.DataFrame(dtype="Int64")
            tournamentPlayedDF = pd.DataFrame(dtype="Int64")

        # Store as attributes
        self.tournamentWonDF = tournamentWonDF
        self.tournamentPlayedDF = tournamentPlayedDF

        for file in files:
            # Add nan row and column
            if file not in self.tournamentWonDF.columns:
                self.tournamentWonDF[file] = np.nan
                self.tournamentWonDF.loc[file] = np.nan
                self.tournamentPlayedDF[file] = np.nan
                self.tournamentPlayedDF.loc[file] = np.nan

                # Use ints
                self.tournamentWonDF = self.tournamentWonDF.astype("Int64")
                self.tournamentPlayedDF = self.tournamentPlayedDF.astype("Int64")

            # Iterate over all columns
            # Against itself, it should be 0.5
            for opponent in self.tournamentWonDF.columns:

                if not pd.isna(self.tournamentWonDF.at[file, opponent]):
                    # If we already have the result, we skip it
                    continue

                print(f"Playing battles between {file} and {opponent}...", flush=True)
                self.playBattles(file, opponent, nBattles=10)

        self.tournamentWonDF = sortMatrix(self.tournamentWonDF)
        self.tournamentPlayedDF = sortMatrix(self.tournamentPlayedDF)

        # Calculate the comparisons matrix for the battles
        self.tournamentDF = sortMatrix(
            self.tournamentWonDF / self.tournamentPlayedDF
        ).astype("float64")

        # Graph the tournament matrix (Too big)
        # self.graphHeatmap(self.tournamentDF, "Battles All")

    def infiniteTournament(self) -> None:
        """
        Play the combinations of players with the most uncertain results.

        Args:
            - None

        Returns:
            - None
        """
        continuePlaying = True
        while continuePlaying:

            low, high = proportion_confint(
                self.tournamentWonDF.values,
                self.tournamentPlayedDF.values,
                alpha=0.05,
                method="wilson",
            )
            uncertaintyDF = pd.DataFrame(
                high - low,
                index=self.tournamentWonDF.index,
                columns=self.tournamentWonDF.columns,
            )

            # Diagonal should be max between current and 0.5-percentage
            for row in uncertaintyDF.index:
                uncertaintyDF.at[row, row] = min(
                    uncertaintyDF.at[row, row],
                    abs(
                        0.5
                        - self.tournamentWonDF.at[row, row]
                        / self.tournamentPlayedDF.at[row, row]
                    ),
                )

            # Graph the uncertainty matrix (Too big)
            # self.graphHeatmap(uncertaintyDF, "Uncertainty")

            # Play battles for the most uncertain pair of players
            mostUncertain = uncertaintyDF.stack().idxmax()

            if uncertaintyDF.at[mostUncertain] < self.maxUncertainty:
                print(
                    "Maximum uncertainty reached. Stopping infinite tournament.",
                    flush=True,
                )
                continuePlaying = False

            else:
                print(
                    mostUncertain[0],
                    "vs",
                    mostUncertain[1],
                    "with",
                    f"{uncertaintyDF.at[mostUncertain]:.2%}",
                    flush=True,
                )

                self.playBattles(
                    mostUncertain[0],
                    mostUncertain[1],
                    nBattles=100 - self.tournamentPlayedDF.at[mostUncertain] % 100,
                )

    def calculateBestParameters(
        self,
        relativeMatrix: pd.DataFrame,
        name: str = "",
        bt: bool = False,
    ) -> dict:
        """
        Calculate the best parameters for each variable in the experiments.csv file.

        Args:
            - relativeMatrix (pd.DataFrame): The matrix with the relative quality of each pair of files.
                It should be skew-symmetric.
            - name (str): The name to use for the graphs.
            - bt (bool): Whether to use Bradley-Terry values.

        Returns:
            - dict: A dictionary with the best parameters for each variable
        """
        # Only use non-random experiments
        nonRandomExperiments = self.experimentData[
            ~self.experimentData["fileName"].str.contains("random")
        ].copy()

        # Keep only those that are in the relativeMatrix
        nonRandomExperiments = nonRandomExperiments[
            nonRandomExperiments["fileName"].isin(relativeMatrix.index)
        ].copy()

        bestParameters = {}

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

                            if bt:
                                # Use BT skill difference: positive means row1 is stronger
                                diff = (
                                    self.bradleyTerryDF.set_index("model").at[
                                        row1["fileName"], "skill"
                                    ]
                                    - self.bradleyTerryDF.set_index("model").at[
                                        row2["fileName"], "skill"
                                    ]
                                )
                                value1 = diff
                                value2 = -diff
                            else:
                                value1 = relativeMatrix.loc[
                                    row1["fileName"], row2["fileName"]
                                ]
                                value2 = relativeMatrix.loc[
                                    row2["fileName"], row1["fileName"]
                                ]

                            parametersDF.loc[row1[column], row2[column]].append(value1)
                            parametersDF.loc[row2[column], row1[column]].append(value2)

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

                # Graph the parameters
                self.graphHeatmap(parametersDF, f"{name} {column}")

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

            # Sort the ranking by value
            ranking = ranking.sort_values(ascending=False)

            bestParameters[column] = ranking

        # Save it to a json file
        with open(
            os.path.join(self.graphDirectory, f"bestParameters{name}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {key: value.to_dict() for key, value in bestParameters.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

        return bestParameters

    def surrogateBestParameters(self) -> dict:
        """
        Calculate the best parameters for the surrogate data.

        We can't reuse the calculateBestParameters function because
        the square matrix would be too big.

        Args:
            - None

        Returns:
            - dict: A dictionary with the best parameters for each variable
        """
        bestParameters = {}

        for column in tqdm.tqdm(
            self.surrogateDF.columns.difference(self.INVALID_COLUMNS + ["score"]),
            desc="Surrogate Columns",
        ):

            parameterValues = self.surrogateDF[column].unique()
            parametersDF = pd.DataFrame(
                [[[] for _ in parameterValues] for _ in parameterValues],
                index=parameterValues,
                columns=parameterValues,
            )

            # Group rows that are identical except for `column` and the invalid columns
            grouped = self.surrogateDF.groupby(
                list(
                    self.surrogateDF.columns.difference(
                        [column] + self.INVALID_COLUMNS + ["score"]
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

                            value1 = row1["score"] - row2["score"]
                            value2 = -value1

                            parametersDF.loc[row1[column], row2[column]].append(value1)
                            parametersDF.loc[row2[column], row1[column]].append(value2)

            # Remove rows and columns with all empty lists
            parametersDF = parametersDF[
                parametersDF.map(lambda x: len(x) > 0).any(axis=1)
            ]
            parametersDF = parametersDF.loc[
                :, (parametersDF.map(lambda x: len(x) > 0)).any(axis=0)
            ]

            # Average the values in each cell or np.nan
            parametersDF = parametersDF.map(
                lambda x: np.mean(x) if len(x) > 0 else np.nan
            )

            # Graph the parameters
            self.graphHeatmap(parametersDF, f"Surrogate {column}")

            # Least-squares score method
            n = parametersDF.shape[0]
            A = np.eye(n) - np.ones((n, n)) / n
            b = parametersDF.sum(axis=1).values
            x_ls = np.linalg.lstsq(A, b, rcond=None)[0]

            ranking = pd.Series(x_ls, index=parametersDF.index).sort_values(
                ascending=False
            )
            ranking = ranking.reindex(parameterValues, fill_value=1)

            # Add missing values as 1 to make sure they appear more
            ranking = ranking.reindex(parameterValues, fill_value=1)

            # Apply softmax to get probabilities
            ranking = np.exp(ranking) / np.sum(np.exp(ranking))

            # Sort the ranking by value
            ranking = ranking.sort_values(ascending=False)

            bestParameters[column] = ranking

        # Save it to a json file
        with open(
            os.path.join(self.graphDirectory, "bestParametersSurrogate.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {key: value.to_dict() for key, value in bestParameters.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

        return bestParameters

    def bradleyTerry(self, winsDF: pd.DataFrame, text: str = "") -> pd.DataFrame:
        """
        Calculate the Bradley-Terry model for the tournament
        results to get a skill rating for each player.

        Args:
            - winsDF (pd.DataFrame): A DataFrame containing the tournament results.
            - text (str): A string to append to the filename for identification.

        Returns:
            - pd.DataFrame: A DataFrame containing the skill ratings for each player.
        """
        players = list(winsDF.columns)

        X = []
        y = []
        weights = []

        for i in range(len(players)):

            for j in range(i + 1, len(players)):

                winsI = int(winsDF.at[players[i], players[j]])
                winsJ = int(winsDF.at[players[j], players[i]])

                vec = np.zeros(len(players))
                vec[i] = 1
                vec[j] = -1

                if winsI > 0:
                    X.append(vec.copy())
                    y.append(1)
                    weights.append(winsI)

                if winsJ > 0:
                    X.append(vec.copy())
                    y.append(0)
                    weights.append(winsJ)

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights, dtype=float)

        model = LogisticRegression(fit_intercept=False)
        model.fit(X, y, sample_weight=weights)

        bradleyTerryDF = pd.DataFrame(
            {"model": players, "skill": model.coef_[0]}
        ).sort_values("skill", ascending=False)

        # Save it to a json file
        with open(
            os.path.join(self.graphDirectory, f"bradleyTerry{text}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                bradleyTerryDF.set_index("model")["skill"].to_dict(),
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

        return bradleyTerryDF

    def createSurrogateModel(self) -> None:
        """
        Create a Gradient Boosted Tree surrogate model to predict the Bradley-Terry
        skill of each player according to the parameters in the experiments.csv file.

        Args:
            - None

        Returns:
            - None
        """
        # Only use non-random experiments
        nonRandomExperiments = self.experimentData[
            ~self.experimentData["fileName"].str.contains("random")
        ].copy()

        # Keep only those that are in the Bradley-Terry model
        nonRandomExperiments = nonRandomExperiments[
            nonRandomExperiments["fileName"].isin(self.bradleyTerryDF["model"])
        ].copy()

        # Need to one-hot encode all columns except the drop columns
        X = nonRandomExperiments.drop(
            ["fileName"] + self.INVALID_COLUMNS, axis=1
        ).copy()
        X = pd.get_dummies(
            X, columns=[col for col in X.columns if col not in self.INVALID_COLUMNS]
        )
        y = nonRandomExperiments["fileName"].map(
            self.bradleyTerryDF.set_index("model")["skill"]
        )

        model = GradientBoostingRegressor()
        model.fit(X, y)

        # Keep the training feature layout so inference rows can be aligned.
        self.surrogateFeatureColumns = X.columns
        self.surrogateModel = model

    def surrogateData(self) -> None:
        """
        Create a surrogate dataset by sampling combinations of parameters,
        predicting their Bradley-Terry skill with the surrogate model.

        Only 10% of the possible combinations are sampled, but the old
        ones are kept in a file.

        Args:
            - None

        Returns:
            - None
        """
        self.surrogateDF = self.experimentData[
            ~self.experimentData["fileName"].str.contains("random")
        ].copy()

        # Add score column
        self.surrogateDF["score"] = self.surrogateDF["fileName"].map(
            self.bradleyTerryDF.set_index("model")["skill"]
        )
        self.surrogateDF.drop(self.INVALID_COLUMNS, axis=1, inplace=True)
        usedCombinations = set(
            tuple(row.drop("score").values) for _, row in self.surrogateDF.iterrows()
        )

        # Create the sets of possible values for each column
        columnValues = {}
        for column in self.surrogateDF.columns:
            if column == "nTeams":
                columnValues[column] = ["inf"]
            elif column != "score":
                columnValues[column] = self.surrogateDF[column].unique()

        # Caculate the max number of combinations
        maxCombinations = np.prod([len(values) for values in columnValues.values()])

        # Obtain 20% of the combinations randomly
        nCombinations = int(maxCombinations * 0.1)
        combinations = set()
        for _ in range(nCombinations):
            combination = tuple(
                np.random.choice(columnValues[column]) for column in columnValues
            )
            if combination not in usedCombinations:

                # Check that useRandom and useMaxDamage are not both "False"
                if not (
                    combination[self.surrogateDF.columns.get_loc("useRandom")]
                    == "False"
                    and combination[self.surrogateDF.columns.get_loc("useMaxDamage")]
                    == "False"
                ):
                    combinations.add(combination)

        # Calculate the score for each combination
        newRows = pd.DataFrame(
            list(combinations),
            columns=[col for col in self.surrogateDF.columns if col != "score"],
        )
        X = pd.get_dummies(newRows)
        X = X.reindex(columns=self.surrogateFeatureColumns, fill_value=0)
        newRows["score"] = self.surrogateModel.predict(X)

        self.surrogateDF = pd.concat(
            [self.surrogateDF, newRows],
            ignore_index=True,
        )

        # Write to in-memory "file"
        buffer = io.StringIO()

        self.surrogateDF.to_csv(buffer, index=False)

        # Move back to the beginning before reading
        buffer.seek(0)

        # Read from the same buffer
        self.surrogateDF = pd.read_csv(buffer)
        combinations = set(
            tuple(row.drop("score").values) for _, row in self.surrogateDF.iterrows()
        )

        # Add the old data
        if os.path.exists(os.path.join(self.dataDirectory, "surrogateData.csv")):
            oldSurrogateDF = pd.read_csv(
                os.path.join(self.dataDirectory, "surrogateData.csv")
            )

            # Only add rows that are not already in the surrogateDF
            oldCombinations = set(
                tuple(row.drop("score").values) for _, row in oldSurrogateDF.iterrows()
            )
            oldCombinations = oldCombinations.difference(
                usedCombinations.union(combinations)
            )

            oldSurrogateDF = oldSurrogateDF[
                oldSurrogateDF.apply(
                    lambda row: tuple(row.drop("score").values) in oldCombinations,
                    axis=1,
                )
            ]

            self.surrogateDF = pd.concat(
                [self.surrogateDF, oldSurrogateDF],
                ignore_index=True,
            )

        # Reorder columns, sort and save
        cols = ["score"] + [col for col in self.surrogateDF.columns if col != "score"]
        self.surrogateDF = self.surrogateDF[cols]
        self.surrogateDF["score"] = self.surrogateDF["score"].astype(float)
        self.surrogateDF = self.surrogateDF.sort_values("score", ascending=False)
        self.surrogateDF.to_csv(
            os.path.join(self.dataDirectory, "surrogateData.csv"), index=False
        )

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

        # Get the files that have better Bradley-Terry skill than random
        worseThanRandom = True
        currentBradleyTerryDF = self.bradleyTerryDF.copy()
        counterR = 1

        while worseThanRandom:

            better = {"random"}
            randomSkill = currentBradleyTerryDF.set_index("model").at["random", "skill"]

            # No longer using the confidence interval
            for opponent in currentBradleyTerryDF["model"]:
                skill = currentBradleyTerryDF.set_index("model").at[opponent, "skill"]

                if skill > randomSkill:
                    better.add(opponent)

            betterDF = sortMatrix(self.tournamentDF.loc[list(better), list(better)])
            self.graphHeatmap(betterDF, f"Battles All Random {counterR}")

            # Recalculate Bradley-Terry with only the better players
            currentBradleyTerryDF = self.bradleyTerry(
                self.tournamentWonDF.loc[list(better), list(better)], text=f"{counterR}"
            )

            # Until random is the worst
            worseThanRandom = currentBradleyTerryDF.iloc[-1]["model"] != "random"
            counterR += 1

        files = {
            f: self.files[f] for f in better.intersection(self.comparisonsDF.index)
        }

        # Sort the files by bradleyTerry skill
        files = dict(
            sorted(
                files.items(),
                key=lambda item: self.bradleyTerryDF.set_index("model").at[
                    item[0], "skill"
                ],
                reverse=True,
            )
        )

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
                smoothed = (
                    df["criticLosses"].rolling(window=windowSize, center=True).mean()
                )
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
                fig.add_trace(
                    go.Scatter(y=smoothed, mode="lines", name=f"{name} Average Rewards")
                )
                if "averageCriticRewards" in df.columns:
                    smoothed = (
                        df["averageCriticRewards"]
                        .rolling(window=windowSize, center=True)
                        .mean()
                    )
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
        Obtain a new experiment by taking the next row from the surrogate dataset.

        Each call returns the next row sequentially, starting from the first.

        Args:
            - None

        Returns:
            - dict: A dictionary with the sampled parameters
        """
        row = self.surrogateDF.iloc[self.surrogateRowIndex].drop("score")
        self.surrogateRowIndex += 1
        return row.to_dict()

    def createNewExperiments(self, n: int = 5) -> None:
        """
        Create n new experiments by sampling from the best parameters.

        Args:
            - n (int): The number of new experiments to create

        Returns:
            - None
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
        allExperiments["nInputs"] = allExperiments["nInputs"].astype("Int64")

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
    dataDirectory = os.path.join(currentDirectory, "data", "experiments")
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


def naturalKey(s):
    """
    A key function for natural sorting of strings containing numbers.

    Args:
        - s (str): The string to generate the key for.

    Returns:
        - list: A list of integers and strings that can be used for natural sorting.
    """
    return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", s)]


def sortMatrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the rows and columns of the matrix in ascending order.

    Args:
        - matrix (pd.DataFrame): The matrix to sort

    Returns:
        - pd.DataFrame: The sorted matrix
    """
    parameterValues = matrix.index.astype(str).tolist()

    # Check if all parameter values are numeric
    allNumeric = all(
        (x in ["inf", "-inf"]) or not np.isnan(pd.to_numeric(x, errors="coerce"))
        for x in parameterValues
    )

    # Sort rows in ascending order
    if allNumeric:
        matrix = matrix.sort_index(key=lambda idx: [float(x) for x in idx])
    else:
        matrix = matrix.sort_index(key=lambda idx: [naturalKey(str(x)) for x in idx])

    # Sort columns in ascending order
    matrix = matrix.reindex(columns=matrix.index)

    return matrix


if __name__ == "__main__":
    logger = MetricsLogger(surrogateGraphs=True)
    logger.graphAllExperiments()
    logger.graphVictoryPercentage()
