from poke_env import ShowdownServerConfiguration, AccountConfiguration
from pokemons import Pokemon08
from players import AIPlayerS
from actors import ActorNetwork03
from moves import Move07
import torch
import asyncio
import appSecrets
import os


async def main():
    # We create a random player
    account_config = AccountConfiguration(
        appSecrets.getShowdownUsername(), appSecrets.getShowdownPassword()
    )

    testPlayer = AIPlayerS(
        network="BlaBlaBla",
        battle_format="gen9randombattle",
        pokemonFeatureExtractor=Pokemon08(Move07),
        account_configuration=None,
        server_configuration=ShowdownServerConfiguration,
    )

    currentDirectory = os.path.dirname(os.path.abspath(__file__))

    actor = ActorNetwork03(testPlayer)
    actor.load_state_dict(
        torch.load(os.path.join(currentDirectory, "data", "experiment88Actor.pth"))
    )
    actor.eval()

    del testPlayer

    player = AIPlayerS(
        network=actor,
        battle_format="gen9randombattle",
        pokemonFeatureExtractor=Pokemon08(Move07),
        account_configuration=account_config,
        server_configuration=ShowdownServerConfiguration,
    )

    if True:
        await player.accept_challenges(opponent=None, n_challenges=20)
    else:
        for _ in range(5):
            # Playing games on the ladder
            await player.ladder(1)

            await player.accept_challenges(opponent=None, n_challenges=1)

            # Print the rating of the player and its opponent after each battle
            for battle in player.battles.values():
                print(battle.rating)

                # Store the rating in a file
                with open("rating.txt", "a") as f:
                    f.write(f"{battle.rating}\n")

            player.reset_battles()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
