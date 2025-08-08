from poke_env.battle import AbstractBattle, Battle, DoubleBattle
from poke_env.player.battle_order import SingleBattleOrder
from poke_env.player.player import Player
import random


class CustomRandomPlayer(Player):
    """
    This class defines a random player baseline

    It can work in any battle format
    It will never attack its partner
    It will always mega evolve if possible
    """

    enablePrint = False
    separator = "-" * 80

    chanceMega = 1
    chanceZ = 0.8
    chanceDynamax = 0.1
    chanceTera = 0.1

    def choose_move(self, battle: AbstractBattle) -> SingleBattleOrder:
        """
        This is the most important method because it must be implemented

        Args:
            battle (AbstractBattle): The current battle

        Returns:
            SingleBattleOrder: The move to be executed
        """

        if isinstance(battle, Battle):
            move = self.chooseRandomSinglesMove(battle)
        elif isinstance(battle, DoubleBattle):
            move = self.choose_random_doubles_move(battle)
        else:
            raise ValueError(
                "battle should be Battle or DoubleBattle. Received %d" % (type(battle))
            )
        if self.enablePrint:
            print(self.separator)
        return move

    def chooseRandomSinglesMove(self, battle: Battle) -> SingleBattleOrder:
        """
        This method chooses a random move in a singles battle

        Args:
            battle (Battle): The current battle

        Returns:
            SingleBattleOrder: The move to be executed
        """
        available_orders = [SingleBattleOrder(move) for move in battle.available_moves]
        # print(battle.opponent_team)

        if self.enablePrint:
            moves = "\t".join(
                order.order.id.capitalize().ljust(10) for order in available_orders
            )
            print(f"Moves: {moves}")

        if battle.can_mega_evolve:
            # Here it megaevolves
            if self.enablePrint:
                print("Pokemon can mega evolve")
            if random.random() < self.chanceMega:
                for order in available_orders:
                    order.mega = True

        if battle.can_z_move and battle.active_pokemon:
            # Here it z-moves
            if self.enablePrint:
                moves = "\t".join(
                    SingleBattleOrder(order, z_move=True)
                    .order.id.capitalize()
                    .ljust(10)
                    for order in set(battle.active_pokemon.available_z_moves)
                )
                print(f"Pokemon can z move: {moves}")
            if random.random() < self.chanceZ:
                available_z_moves = set(battle.active_pokemon.available_z_moves)
                available_orders = [
                    SingleBattleOrder(move, z_move=move in available_z_moves)
                    for move in battle.available_moves
                ]

        if battle.can_dynamax:
            # Here it dynamaxes
            if self.enablePrint:
                print("Pokemon can dynamax")
            if random.random() < self.chanceDynamax:
                for order in available_orders:
                    order.dynamax = True

        if battle.can_tera:
            # Here it terastallizes
            if self.enablePrint:
                print(f"Pokemon can terastallize: {battle.can_tera.name}")
            if random.random() < self.chanceTera:
                for order in available_orders:
                    order.terastallize = True

        if self.enablePrint:
            switches = "\t".join(
                switch.species.ljust(10) for switch in battle.available_switches
            )
            print(f"Switches: {switches}")

        if battle.available_switches:
            if random.random() < 1 / (1 + len(available_orders)):
                # I will randomly choose between all the moves and switching
                available_orders = [
                    SingleBattleOrder(switch) for switch in battle.available_switches
                ]

        if available_orders:
            move = random.choice(available_orders)
            if self.enablePrint:
                print(f"Option that was chosen: {str(move)[8:]}")
            return move
        else:
            return self.choose_default_move()
