from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player import RandomPlayer
import serverControl
import asyncio
import warnings
import logging
import os

# Set up logging to capture warnings into a file
logging.basicConfig(
    filename="warnings.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Redirect warnings to the log file
def log_warning(message, category, filename, lineno, file=None, line=None):
    print(category.__name__)
    logging.warning(f"{category.__name__}: {message} (in {filename} at line {lineno})")


warnings.showwarning = log_warning


async def main():
    serverControl.startServer()

    serverConfig = ServerConfiguration(
        f"ws://localhost:{int(os.getenv("SERVER_PORT"))}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    # We create a random player
    player = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=serverConfig,
    )

    # We create another random player
    second_player = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=serverConfig,
    )

    # The battle_against method initiates a battle between two players.
    # Here we are using asynchronous programming (await) to start the battle.
    await player.battle_against(second_player, n_battles=1000)

    # We can now print the results of the battles
    print(
        f"Player {player.username} won {player.n_won_battles} out of {player.n_finished_battles} played"
    )
    print(
        f"Player {second_player.username} won {second_player.n_won_battles} out of {second_player.n_finished_battles} played"
    )


if __name__ == "__main__":
    asyncio.run(main())
