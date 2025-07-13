from poke_env.environment import AbstractBattle, Pokemon, Move
from players.AbstractAIPlayer import AbstractAIPlayer
import numpy as np


class AIPlayerS(AbstractAIPlayer):
    """
    This will have all the information
    """

    N_F_TYPES = 20  # Tera stellar and Pawmot
    N_F_MOVE = AbstractAIPlayer.encoder.NUM_UNIQUE_MOVES + N_F_TYPES + 3 + 1 + 1
    N_F_POKEMON = (
        AbstractAIPlayer.encoder.NUM_UNIQUE_FORMS
        + N_F_TYPES
        + N_F_TYPES
        + 1
        + 1
        + N_F_TYPES
        + 1
        + 6
        + (1 + AbstractAIPlayer.encoder.NUM_UNIQUE_ABILITIES)
        + 1
        + (2 * 6)
        + 7
        + (1 + AbstractAIPlayer.encoder.NUM_UNIQUE_ITEMS)
        + 4 * (1 + N_F_MOVE)
    )
    N_F_TOTAL = 2 + (1 + N_F_POKEMON) * 12

    N_OUTPUTS = 14  # 8 moves + 6 switches

    def getInputs(self, battle: AbstractBattle) -> list[float]:
        """
        This method will return the inputs for the neural network

        The feature vector will have:
            - The index of the active pokemon in the team
            - The player's team (1 + {self.N_F_POKEMON} features per pokemon):
                - Presence indicator (1 if the pokemon is present, 0 otherwise)
                - The encoding of the pokemon (see encodePokemon)
            - The index of the opponent's active pokemon in the opponent's team
            - The opponent's team (1 + {self.N_F_POKEMON} features per pokemon):
                - Presence indicator (1 if the pokemon is present, 0 otherwise)
                - The encoding of the pokemon (see encodePokemon)

        Args:
            battle (AbstractBattle): The current battle

        Returns:
            list: The inputs for the neural network
        """
        inputs = []

        # Which pokemon is active
        try:
            index = list(battle.team.values()).index(battle.active_pokemon) + 1
        except ValueError:
            index = 0
        inputs.append(index)

        # First our team
        for pokemon in battle.team.values():
            inputs += [1] + self.encodePokemon(pokemon)
        # Fill with zeros if unknown pokemon
        inputs += (6 - len(battle.team)) * ([0] + [0.0] * self.N_F_POKEMON)

        # Which opponent's pokemon is active
        try:
            index = (
                list(battle.opponent_team.values()).index(
                    battle.opponent_active_pokemon
                )
                + 1
            )
        except ValueError:
            index = 0
        inputs.append(index)

        # The opponent's team
        for pokemon in battle.opponent_team.values():
            inputs += [1] + self.encodePokemon(pokemon)
        # Fill with zeros if unknown pokemon
        inputs += (6 - len(battle.opponent_team)) * ([0] + [0.0] * self.N_F_POKEMON)

        return inputs

    def encodePokemon(self, pokemon: Pokemon) -> list[float]:
        """
        This method will encode a pokemon into a feature vector

        The feature vector will have:
            - The form of the pokemon (encoded as a list of encoder.NUM_UNIQUE_FORMS integers)
            - The original type of the pokemon (encoded as a list of {self.N_F_TYPES} integers)
            - The current types of the pokemon (encoded as a list of {self.N_F_TYPES} integers)
            - If the pokemon has already tera'd (encoded as an integer)
            - If the tera type is known (encoded as an integer)
            - The tera type of the pokemon (encoded as a list of {self.N_F_TYPES} integers)
            - The level of the pokemon as a float
            - The base stats of the pokemon (list of 6 floats)
            - The ability of the pokemon:
                - The presence indicator of the ability (1 if known, 0 otherwise)
                - The ability (encoded as a list of encoder.NUM_UNIQUE_ABILITIES integers)
            - The HP fraction of the pokemon
            - The 6 stats of the pokemon:
                - The presence indicator of the stat (1 if known, 0 otherwise)
                - The value of the stat (a float)
            - The stat boosts of the pokemon (list of 7 floats)
            - The item:
                - The presence indicator of the item (1 if known, 0 otherwise)
                - The item (encoded as a list of encoder.NUM_UNIQUE_ITEMS integers)
            - The 4 moves of the pokemon:
                - The presence indicator of the move
                - The encoded move (list of self.N_F_MOVE integers)

        Args:
            - pokemon (Pokemon): The pokemon to be encoded

        Returns:
            - list[float]: The feature vector of the pokemon
        """
        featureVector = []

        # The form of the pokemon
        featureVector.extend(
            self.encoder.encodeFormList(pokemon.species, pokemon.base_species)
        )

        # The original type of the pokemon
        types = [0] * self.N_F_TYPES
        for t in pokemon.original_types:
            types[t.value - 1] = 1
        featureVector += types

        # The current pokemon type
        cTypes = [0] * self.N_F_TYPES
        for t in pokemon.types:
            cTypes[t.value - 1] = 1
        featureVector += cTypes

        # If the pokemon has already tera'd
        featureVector.append(int(pokemon.is_terastallized))

        # If the tera type is known
        if pokemon.tera_type:
            tTypes = [0] * self.N_F_TYPES
            tTypes[pokemon.tera_type.value - 1] = 1
            featureVector += [1] + tTypes
        else:
            # If the tera type is not known we add a zero
            featureVector += [0] + ([0] * self.N_F_TYPES)

        # Add the level
        featureVector.append(pokemon.level / 100)

        # Add the base stats
        for stat in self.STATS:
            featureVector.append(pokemon.base_stats[stat] / 255)

        # Add the ability
        if pokemon.ability is not None:
            # We encode the ability
            featureVector.append(1)
            featureVector.extend(self.encoder.encodeAbilityList([pokemon.ability]))
        else:
            # If the ability is not known we add a zero
            featureVector.append(0)
            # We encode the possible abilities
            featureVector.extend(
                self.encoder.encodeAbilityList(pokemon.possible_abilities)
            )

        # The HP fraction of the pokemon
        featureVector.append(pokemon.current_hp_fraction)

        # The stats
        for stat in self.STATS:
            if pokemon.stats[stat] is not None:
                featureVector.append(1)
                featureVector.append(np.log(pokemon.stats[stat] + 1) / np.log(1000))
            else:
                featureVector.extend([0, 0])

        # Stat boosts
        for stat in [s for s in self.STATS if s != "hp"] + ["accuracy", "evasion"]:
            featureVector.append(pokemon.boosts[stat] / 6)

        # Add the item
        if pokemon.item and pokemon.item != "unknown_item":
            # We encode the item
            featureVector.append(1)
            featureVector.extend(self.encoder.encodeItemList(pokemon.item))
        else:
            # If the item is not known we add a zero
            featureVector.extend([0] * (1 + self.encoder.NUM_UNIQUE_ITEMS))

        # The moves
        moves = []
        for move in pokemon.moves.values():
            # We encode the move
            moves.append(1)
            moves.extend(self.encodeMove(move))
        # We fill with zeros if the pokemon has less than 4 moves
        moves += [0] * (1 + self.N_F_MOVE) * (4 - len(pokemon.moves.values()))
        featureVector += moves

        return featureVector

    def encodeMove(self, move: Move) -> list[float]:
        """
        This method will encode a move into a feature vector

        The feature vector will have:
            - The name of the move (encoded as a list of encoder.NUM_UNIQUE_MOVES integers)
            - The type of the move (encoded as a list of {self.N_F_TYPES} integers)
            - The category of the move (encoded as a list of 3 integers, one for each category)
                - PHYSICAL
                - SPECIAL
                - STATUS
            - The base power of the move (normalized to a float between 0 and 1)
                Divided by 120 because higher than that and they become very rare.
            - The accuracy of the move (a float between 0 and 1)
                It is already given between 0 and 1.

        Args:
            move (Move): The move to be encoded

        Returns:
            list[float]: The feature vector of the move
        """
        toret = []
        toret.extend(self.encoder.encodeMoveList(move.id))

        types = [0] * self.N_F_TYPES
        types[move.type.value - 1] = 1
        toret += types

        # Add the category of the move
        moveCategory = [0] * 3
        moveCategory[move.category.value - 1] = 1
        toret.extend(moveCategory)

        # Add the base power of the move
        toret.append(move.base_power / 120)

        # Add the accuracy of the move
        toret.append(move.accuracy)

        return toret


if __name__ == "__main__":
    print(f"Number of features: {AIPlayerS.N_F_TOTAL}")
