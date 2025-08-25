from pokemons import AbstractPokemon
import numpy as np


class Pokemon06(AbstractPokemon):

    N_F_POKEMON = (
        1
        + 1
        + 1
        + 1
        + AbstractPokemon.encoder.N_F_TYPES
        + AbstractPokemon.encoder.N_F_TYPES
        + 1
        + len(AbstractPokemon.encoder.STATS)
        + (1 + 1)
        + 1
        + 1
        + (2 * len(AbstractPokemon.encoder.STATS))
        + len(AbstractPokemon.encoder.BOOSTABLE_STATS)
        + (1 + 1)
        + 1
        + 4
    )

    def getFeatures(self, pokemon, battle) -> list[float]:
        """
        This method will encode a pokemon into a feature vector

        The feature vector will have:
            - If the pokemon has been revealed (boolean)
            - If the pokemon is currently active (boolean)
            - If the pokemon has fainted (boolean)
            - The form of the pokemon (encoded as an integer)
            - The original type of the pokemon (encoded as a list of {self.encoder.N_F_TYPES} integers)
            - The current types of the pokemon (encoded as a list of {self.encoder.N_F_TYPES} integers)
            - The level of the pokemon as a float
            - The base stats of the pokemon (list of {self.encoder.N_F_STATS} floats)
            - The ability's presence indicator
            - The ability of the pokemon (encoded as an integer)
            - If it is the first turn of the pokemon (boolean)
            - The HP fraction of the pokemon
            - The {self.encoder.N_F_STATS} stats of the pokemon:
                - The presence indicator of the stat (1 if known, 0 otherwise)
                - The value of the stat (a float) in logarithmic scale
                When the hp is unknown we use the max_hp. They are always the same
                value except for Magnezone (seems to be an error)
            - The stat boosts of the pokemon (list of {self.encoder.N_F_BOOSTABLE_STATS} floats)
            - The item's presence indicator
            - The item of the pokemon (encoded as an integer)
            - Protect counter (integer)
            - The 4 moves of the pokemon:
                - The presence indicator of the move
                - The encoded move

        Args:
            - pokemon (Pokemon): The pokemon to be encoded

        Returns:
            - list[float]: The feature vector of the pokemon
        """
        featureVector = []

        # If the pokemon has been revealed
        featureVector.append(int(pokemon.revealed))

        # If the pokemon is currently active
        featureVector.append(
            int(pokemon.active and pokemon in battle.all_active_pokemons)
        )

        # If the pokemon has fainted
        featureVector.append(int(pokemon.fainted))

        # The form of the pokemon
        form = self.encoder.encodeForm(pokemon.species)
        if form == -1:
            # It was a cosmetic form
            form = self.encoder.encodeForm(pokemon.base_species)
        featureVector.append(form)

        # The original type of the pokemon
        types = [0] * AbstractPokemon.encoder.N_F_TYPES
        for t in pokemon.original_types:
            types[t.value - 1] = 1
        featureVector += types

        # The current pokemon type
        cTypes = [0] * AbstractPokemon.encoder.N_F_TYPES
        for t in pokemon.types:
            cTypes[t.value - 1] = 1
        featureVector += cTypes

        # Add the level
        featureVector.append(pokemon.level / 100)

        # Add the base stats
        for stat in AbstractPokemon.encoder.STATS:
            featureVector.append(pokemon.base_stats[stat] / 255)

        # Add the ability
        ability = self.encoder.encodeAbility(pokemon.ability)
        if ability == -1:
            # We don't know the ability
            featureVector.extend([0, 0])
        else:
            featureVector.extend([1, ability])

        # If it is the first turn of the pokemon
        featureVector.append(int(pokemon.first_turn))

        # The HP fraction of the pokemon
        featureVector.append(pokemon.current_hp_fraction)

        # The stats
        for stat in AbstractPokemon.encoder.STATS:
            if pokemon.stats[stat] is not None:
                featureVector.append(1)
                featureVector.append(np.log(pokemon.stats[stat] + 1) / np.log(1000))
            elif stat == "hp":
                featureVector.append(1)
                featureVector.append(np.log(pokemon.max_hp + 1) / np.log(1000))
            else:
                featureVector.extend([0, 0])

        # Stat boosts
        for stat in AbstractPokemon.encoder.BOOSTABLE_STATS:
            featureVector.append(pokemon.boosts[stat] / 6)

        # Add the item
        item = self.encoder.encodeItem(pokemon.item)
        if item == -1:
            # We don't know the item
            featureVector.extend([0, 0])
        else:
            featureVector.extend([1, item])

        # Protect counter
        if pokemon.protect_counter and pokemon in battle.all_active_pokemons:
            featureVector.append(pokemon.protect_counter)
        else:
            featureVector.append(0)

        # The moves
        moves = []
        for move in pokemon.moves.values():
            # We encode the move
            moves.append(1)
            moves.extend(self.moveEncoder.getFeatures(move))
        # We fill with zeros if the pokemon has less than 4 moves
        moves += (
            [0]
            * (1 + self.moveEncoder.getNumberOfFeatures())
            * (4 - len(pokemon.moves.values()))
        )
        featureVector += moves

        return featureVector
