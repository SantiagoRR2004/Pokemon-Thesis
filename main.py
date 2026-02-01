import os
import pandas as pd
import actors
import critics
import moves
import pokemons
import asyncio
import players
import rewards
import training
import getpass
import subprocess
import shutil

if __name__ == "__main__":

    while True:
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

                finishExperiment = False

                # Until the experiment is finished, keep trying
                while not finishExperiment:
                    try:
                        actor = getattr(actors, row.actor)

                        move = getattr(moves, row.move)

                        pokemon = getattr(pokemons, row.pokemon)

                        rewardsClass = getattr(rewards, row.reward)

                        critic = None
                        if row.TrainingMethod == "actorCritic":
                            critic = getattr(critics, row.critic)

                        args = {
                            "actorClass": actor,
                            "playerClass": getattr(players, row.player),
                            "moveClass": move,
                            "pokemonClass": pokemon,
                            "criticClass": critic,
                            "rewardsClass": rewardsClass,
                        }

                        # Add the rest of the columns as arguments
                        for col in df.columns:
                            if col not in [
                                "actor",
                                "critic",
                                "TrainingMethod",
                                "move",
                                "pokemon",
                                "player",
                                "nInputs",
                                "reward",
                            ]:
                                args[col] = getattr(row, col)

                        t = training.Trainer(**args)

                        asyncio.run(t.main())

                        finishExperiment = True
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        username = getpass.getuser()
                        # Shut down the server forcefully
                        # This will break VSCode if it is running from there
                        subprocess.run(["pkill", "-u", username, "-f", "node"])
                        shutil.rmtree(
                            os.path.join(
                                os.path.dirname(os.path.abspath(__file__)),
                                "pokemon-showdown",
                            )
                        )

        # Create new experiments
        from data import test_Experiments
