import matplotlib.pyplot as plt
from collections import Counter
import serverControl
import otherPlayers
import numpy as np
import asyncio
import json
import os


class RandomVictoryPercentage:
    """
    In theory if a player plays against himself, he should
    only win 50% of the time. This class is created to calculate
    the number of battles needed to be sure that the victory
    percentage is within a percentage (1%)
    """

    def __init__(self, decimalPrecision: int = 2) -> None:
        """
        Initializes the random victory percentage and also
        calculates the percentage needed.

        Args:
            - decimalPrecision (int): The number of decimal places to round the percentage to.

        Returns:
            - None
        """
        self.decimalPrecision = decimalPrecision

        self.currentDirectory = os.path.dirname(os.path.abspath(__file__))
        self.dataFile = os.path.join(self.currentDirectory, "convergenceResults.json")

        # Load the data from the JSON file
        if os.path.exists(self.dataFile):
            with open(self.dataFile, "r") as f:
                self.data = json.load(f)
        else:
            self.data = []

        # Start the server
        self.resetServer()
        self.serverCounter = 0

        try:

            # Infinite loop
            while True:
                # Calculate the percentage needed
                self.calculatePercentage()

                # Save the data to the JSON file
                with open(self.dataFile, "w") as f:
                    f.write("[\n")

                    for row in self.data:
                        json.dump(row, f)
                        f.write(",\n")

                    # Remove the last comma and newline
                    f.seek(f.tell() - 2, os.SEEK_SET)
                    f.write("\n]\n")

                # Percentage graph
                self.plotPercentages()

                # Convergence graph
                self.plotConvergence()

        except KeyboardInterrupt:
            print("Program interrupted by user")

        finally:
            # End the server process
            if getattr(self, "p", None) is not None:
                serverControl.endProcess(self.p)
                self.p.wait()

    def resetServer(self) -> None:
        """
        Resets the server by ending the current process and starting a new one.
        Also creates two new random players.

        Args:
            - None

        Returns:
            - None
        """
        if getattr(self, "p", None) is not None:
            serverControl.endProcess(self.p)
            self.p.wait()

        self.p = serverControl.startServer()

        if getattr(self, "player1", None) is not None:
            del self.player1
        if getattr(self, "player2", None) is not None:
            del self.player2

        # Create the players
        self.player1 = otherPlayers.getRandomPlayer(
            args={"server_configuration": serverControl.getServerConfiguration()}
        )
        self.player2 = otherPlayers.getRandomPlayer(
            args={"server_configuration": serverControl.getServerConfiguration()}
        )

    def calculatePercentage(self) -> None:
        """
        Calculates the percentage needed by playing battles between two random players
        until the percentage is within the desired range.

        Args:
            - None

        Returns:
            - None
        """
        # Add one extra row
        self.data.append([])

        for row in self.data:

            percentage = sum(row) / len(row) if len(row) > 0 else 0

            while len(row) < 100 or round(percentage, self.decimalPrecision) != 0.5:

                # Restart server every 100 battles
                if (self.serverCounter + 1) % 101 == 0:
                    self.resetServer()
                    self.serverCounter = 0

                # Play a single game
                asyncio.run(self.player1.battle_against(self.player2, n_battles=1))
                row.append(int(self.player1.win_rate))

                # Prepare for next iteration
                percentage = sum(row) / len(row)
                self.serverCounter += 1
                self.player1.reset_battles()
                self.player2.reset_battles()

    def plotPercentages(self) -> None:
        """
        Plots the percentages of the battles.

        # For each row it plots the percentage until
        that point, so we can see how the percentage converges to 50%.

        Args:
            - None

        Returns:
            - None
        """
        for row in self.data:
            plt.plot(
                np.arange(1, len(row) + 1), np.cumsum(row) / np.arange(1, len(row) + 1)
            )

        plt.xlabel("Number of Battles")
        plt.ylabel("Victory Percentage")
        plt.title("Convergence of Victory Percentage")
        plt.axhline(y=0.5, color="r", linestyle="--")

        # Save the plot
        plt.savefig(os.path.join(self.currentDirectory, "convergencePlot.png"))
        plt.clf()

    def plotConvergence(self) -> None:
        """
        Plots the convergence of the victory percentage by counting the frequency
        of each number of battles needed to converge to 50%.

        Args:
            - None

        Returns:
            - None
        """
        lengths = [len(row) for row in self.data]
        counts = Counter(lengths)

        # Sort the counts by the number of battles
        x, y = zip(*sorted((k, v) for k, v in counts.items()))

        # Simply plot each point
        plt.scatter(x, y)
        plt.xlabel("Number of Battles")
        plt.ylabel("Frequency")
        plt.title("Convergence of Victory Percentage")

        # Y-axis from 0 to the max
        plt.ylim(0, max(counts.values()) + 1)

        # Save the plot
        plt.savefig(os.path.join(self.currentDirectory, "convergenceFrequency.png"))
        plt.clf()


if __name__ == "__main__":
    RandomVictoryPercentage()
