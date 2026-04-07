import randomTeams.randomTeam as randomTeam
from poke_env.player import RandomPlayer, Player
import serverControl
import metricsLogger
import asyncio
import signal
import time
import os


async def main(player1: Player, player2: Player, nTeams: int) -> None:
    p = serverControl.startServer()

    nEpisodes = 64
    nEpochs = 1000

    victoryPercentage = []
    nTurns = []

    start = time.time()

    for epoch in range(nEpochs):

        if nTeams == float("inf"):
            # We create the AI player
            player = player1(
                battle_format="gen9randombattle",
                max_concurrent_battles=nEpisodes,
                server_configuration=serverControl.getServerConfiguration(),
            )

            # We create another random player
            second_player = player2(
                battle_format="gen9randombattle",
                max_concurrent_battles=nEpisodes,
                server_configuration=serverControl.getServerConfiguration(),
            )
        else:

            # We create the AI player
            player = player1(
                battle_format="gen9purehackmons",
                team=randomTeam.selectRandomTeam(nTeams),
                max_concurrent_battles=nEpisodes,
                server_configuration=serverControl.getServerConfiguration(),
            )

            # We create another random player
            second_player = player2(
                battle_format="gen9purehackmons",
                team=randomTeam.selectRandomTeam(nTeams),
                max_concurrent_battles=nEpisodes,
                server_configuration=serverControl.getServerConfiguration(),
            )

        # Run n_battles (Episodes)
        await player.battle_against(second_player, n_battles=nEpisodes)

        percentage = player.n_won_battles / player.n_finished_battles

        # We can now print the results of the battles
        print(f"{epoch+1:0{len(str(nEpochs))}d}/{nEpochs}", end=" ", flush=True)
        print(
            time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    (time.time() - start) * (nEpochs - (epoch + 1)) / (epoch + 1)
                ),
            ),
            end=" ",
            flush=True,
        )
        print(
            f"{player.username} won {percentage*100:.2f}% of battles.",
            end=" ",
            flush=True,
        )
        print()

        victoryPercentage.append(percentage)
        nTurns.append(
            sum(battle.turn for battle in player.battles.values()) / len(player.battles)
        )

        if (epoch + 1) % 100 == 0 and (epoch + 1) != nEpochs:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            p.wait()
            p = serverControl.startServer()

    # Save the metrics
    kwargs = {
        "victoryPercentage": victoryPercentage,
        "nTurns": nTurns,
    }

    metricsLogger.saveData(**kwargs)


if __name__ == "__main__":

    if os.name == "nt":
        # For Windows
        os.system("cls")
    else:
        # For Linux/macOS
        os.system("clear")

    import ignorePokeEnvProblems

    asyncio.run(main(player1=RandomPlayer, player2=RandomPlayer, nTeams=float("inf")))
