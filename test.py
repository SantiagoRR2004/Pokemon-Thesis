from poke_env import ShowdownServerConfiguration, AccountConfiguration

# from poke_env.player import RandomPlayer
# from poke_env.player.baselines import SimpleHeuristicsPlayer
from MyRandomPlayer import RandomPlayer
import asyncio
import appSecrets


async def main():
    # We create a random player
    account_config = AccountConfiguration(
        appSecrets.getShowdownUsername(), appSecrets.getShowdownPassword()
    )

    player = RandomPlayer(
        account_configuration=account_config,
        server_configuration=ShowdownServerConfiguration,
        battle_format="gen9randombattle",
    )

    # # Sending challenges to 'your_username'
    # await player.send_challenges("your_username", n_challenges=1)

    # # Accepting one challenge from any user
    # await player.accept_challenges(None, 1)

    # # Accepting three challenges from 'your_username'
    # await player.accept_challenges('your_username', 3)

    # Playing 5 games on the ladder
    await player.ladder(1)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
