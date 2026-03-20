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

    player = AIPlayerS(
        network="BlaBlaBla",
        battle_format="gen9randombattle",
        pokemonFeatureExtractor=Pokemon08(Move07),
        account_configuration=account_config,
        server_configuration=ShowdownServerConfiguration,
    )

    currentDirectory = os.path.dirname(os.path.abspath(__file__))

    actor = ActorNetwork03(player)
    actor.load_state_dict(
        torch.load(
            os.path.join(
                currentDirectory, "data", "experiments", "experiment88Actor.pth"
            ),
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )
    actor.eval()

    player.neuralNetwork = actor

    if False:
        await player.accept_challenges(opponent=None, n_challenges=20)
    else:
        for _ in range(100):
            # Playing games on the ladder
            await player.ladder(1)

            for battle in player.battles.values():
                # This is the rating before the battle
                print(battle.rating)

                # Store the rating in a file
                with open("rating.txt", "a") as f:
                    f.write(f"{battle.rating}\n")

            player.reset_battles()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
