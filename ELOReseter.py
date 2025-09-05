from poke_env import ShowdownServerConfiguration, AccountConfiguration
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.player import Player
import appSecrets
import asyncio
import time


class ForfeitPlayer(Player):
    def choose_move(self, battle) -> ForfeitBattleOrder:
        return ForfeitBattleOrder()


async def main():
    # We create a random player
    account_config = AccountConfiguration(
        appSecrets.getShowdownUsername(), appSecrets.getShowdownPassword()
    )

    player = ForfeitPlayer(
        account_configuration=account_config,
        server_configuration=ShowdownServerConfiguration,
        battle_format="gen9randombattle",
    )

    rating = 1001

    while rating > 1000:
        await player.ladder(1)
        for battle in player.battles.values():
            rating = battle.rating
        player.reset_battles()
        time.sleep(5)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
