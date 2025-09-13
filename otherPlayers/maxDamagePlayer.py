from poke_env.player import Player
import random


class MaxRandomDamagePlayer(Player):
    def choose_move(self, battle):
        # Chooses a move with the highest base power when possible
        if battle.available_moves:

            moves = battle.available_moves
            weights = [move.base_power if move.base_power > 0 else 0 for move in moves]

            if sum(weights) != 0:

                # Randomly choose based on base power as weight
                chosen_move = random.choices(moves, weights=weights, k=1)[0]

                # Creating an order for the selected move
                return self.create_order(chosen_move)

            else:
                # If all moves have 0 base power, choose randomly among them
                return self.choose_random_move(battle)
        else:
            # If no attacking move is available, perform a random switch
            # This involves choosing a random move, which could be a switch or another available action
            return self.choose_random_move(battle)
