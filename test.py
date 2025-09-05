from poke_env import ShowdownServerConfiguration, AccountConfiguration
from pokemons import Pokemon00
from players import AIPlayer00
from moves import Move00
from actors import ActorNetwork01
import asyncio
import appSecrets


async def main():
    # We create a random player
    account_config = AccountConfiguration(
        appSecrets.getShowdownUsername(), appSecrets.getShowdownPassword()
    )

    testPlayer = AIPlayer00(
        network="BlaBlaBla",
        pokemonFeatureExtractor=Pokemon00(Move00),
        account_configuration=account_config,
        server_configuration=ShowdownServerConfiguration,
    )

    player = AIPlayer00(
        account_configuration=account_config,
        server_configuration=ShowdownServerConfiguration,
        battle_format="gen9randombattle",
        network=ActorNetwork01(testPlayer),
        pokemonFeatureExtractor=Pokemon00(Move00),
    )

    del testPlayer

    for _ in range(5):
        # Playing games on the ladder
        await player.ladder(0)

        # Print the rating of the player and its opponent after each battle
        for battle in player.battles.values():
            print(battle.rating, battle.opponent_rating)

            # Store the rating in a file
            with open("rating.txt", "a") as f:
                f.write(f"{battle.rating},{battle.opponent_rating}\n")

        player.reset_battles()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
