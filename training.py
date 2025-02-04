from MyRandomPlayer import CustomRandomPlayer
import serverControl
import asyncio


async def main():
    serverControl.startServer()

    # We create a random player
    player = CustomRandomPlayer(
        battle_format="gen9randombattle",
    )

    second_player = CustomRandomPlayer()

    # The battle_against method initiates a battle between two players.
    # Here we are using asynchronous programming (await) to start the battle.
    await player.battle_against(second_player, n_battles=200)

    # We can now print the results of the battles
    print(
        f"Player {player.username} won {player.n_won_battles} out of {player.n_finished_battles} played"
    )
    print(
        f"Player {second_player.username} won {second_player.n_won_battles} out of {second_player.n_finished_battles} played"
    )


if __name__ == "__main__":
    asyncio.run(main())
