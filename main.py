import os
import pandas as pd
import actors
import asyncio
from players import AIPlayer
import training


if __name__ == "__main__":
    dataFile = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "experiments.csv"
    )

    # Open with pandas
    df = pd.read_csv(dataFile)

    for row in df.itertuples():
        # Check if the completePath does not exist
        completePath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            row.fileName + ".parquet",
        )

        if not os.path.exists(completePath):
            print(f"Running experiment {row.fileName}...")
            actor = getattr(actors, row.actor)
            actorI = actor(AIPlayer)
            if row.TrainingMethod == "actorCritic":
                critic = getattr(actors, row.critic)
                criticI = critic(AIPlayer)
            elif row.TrainingMethod == "actor":
                criticI = None
            else:
                criticI = None

            args = {
                "actor": actorI,
                "critic": criticI,
            }

            # Add the rest of the columns as arguments
            for col in df.columns:
                if col not in ["actor", "critic", "TrainingMethod"]:
                    args[col] = getattr(row, col)

            asyncio.run(training.main(**args))
