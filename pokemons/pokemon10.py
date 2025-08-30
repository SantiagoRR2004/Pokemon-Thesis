from poke_env.battle.effect import _VOLATILE_STATUS_EFFECTS
from pokemons import AbstractPokemon
import numpy as np


class Pokemon10(AbstractPokemon):

    N_F_POKEMON = (
        1
        + 1
        + 1
        + 1
        + AbstractPokemon.encoder.N_F_TYPES
        + AbstractPokemon.encoder.N_F_TYPES
        + 1
        + (1 + AbstractPokemon.encoder.N_F_TYPES)
        + 1
        + len(AbstractPokemon.encoder.STATS)
        + 1
        + 3
        + (1 + AbstractPokemon.encoder.NUM_UNIQUE_ABILITIES)
        + 1
        + 1
        + 1
        + (2 * len(AbstractPokemon.encoder.STATS))
        + len(AbstractPokemon.encoder.BOOSTABLE_STATS)
        + (1 + AbstractPokemon.encoder.NUM_UNIQUE_ITEMS)
        + 1
        + len(AbstractPokemon.encoder.STATUS)
        + len(_VOLATILE_STATUS_EFFECTS)
        + (1 + 1 + 1 + 1)
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
            - If the pokemon has already tera'd (boolean)
            - If the tera type is known (boolean)
            - The tera type of the pokemon (encoded as a list of {self.encoder.N_F_TYPES} integers)
            - The level of the pokemon as a float
            - The base stats of the pokemon (list of {self.encoder.N_F_STATS} floats)
            - The weight of the pokemon (float between 0 and 1)
            - The gender of the pokemon (3 One-Hot Encoding)
            - The ability of the pokemon:
                - The presence indicator
                - The ability (encoded as a list of encoder.NUM_UNIQUE_ABILITIES integers)
            - The STAB multiplier of the pokemon (float)
            - If it is the first turn of the pokemon (boolean)
            - The HP fraction of the pokemon
            - The {self.encoder.N_F_STATS} stats of the pokemon:
                - The presence indicator of the stat (1 if known, 0 otherwise)
                - The value of the stat (a float) in logarithmic scale
                When the hp is unknown we use the max_hp. They are always the same
                value except for Magnezone (seems to be an error)
            - The stat boosts of the pokemon (list of {self.encoder.N_F_BOOSTABLE_STATS} floats)
            - The item:
                - The presence indicator of the item (1 if known, 0 otherwise)
                - The item (encoded as a list of encoder.NUM_UNIQUE_ITEMS integers)
            - Protect counter (integer)
            - Non-volatile status ({self.N_F_STATUS} One-Hot Encoding)
                They will be 1 except for toxic and sleep.
                These will be the numbers of turns the status has been active.
            - Volatile status effects ({self.N_F_VOLATILE_STATUS} One-Hot Encoding)
                It uses the max of turns the status has been active and one.
            - Currently in a 2 turn move (4 features):
                - If the pokemon is recharging (boolean)
                - If the pokemon is preparing (boolean)
                - If the pokemon is preparing a move (float between 0 and 1)
                - If the pokemon is preparing a target (boolean)
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

        # If the pokemon has already tera'd
        featureVector.append(int(pokemon.is_terastallized))

        # If the tera type is known
        if pokemon.tera_type:
            tTypes = [0] * AbstractPokemon.encoder.N_F_TYPES
            tTypes[pokemon.tera_type.value - 1] = 1
            featureVector += [1] + tTypes
        else:
            # If the tera type is not known we add a zero
            featureVector += [0] + ([0] * AbstractPokemon.encoder.N_F_TYPES)

        # Add the level
        featureVector.append(pokemon.level / 100)

        # Add the base stats
        for stat in AbstractPokemon.encoder.STATS:
            featureVector.append(pokemon.base_stats[stat] / 255)

        # Add the weight
        featureVector.append(pokemon.weight / 1000)

        # Add the gender
        gender = [0, 0, 0]
        gender[pokemon.gender.value - 1] = 1
        featureVector.extend(gender)

        # Add the ability
        if pokemon.ability is not None:
            # We encode the ability
            featureVector.append(1)
            featureVector.extend(
                AbstractPokemon.encoder.encodeAbilityList([pokemon.ability])
            )
        else:
            # If the ability is not known we add a zero
            featureVector.append(0)
            # We encode the possible abilities
            featureVector.extend(
                AbstractPokemon.encoder.encodeAbilityList(pokemon.possible_abilities)
            )

        # The STAB multiplier
        featureVector.append(pokemon.stab_multiplier)

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
        if pokemon.item and pokemon.item != "unknown_item":
            # We encode the item
            featureVector.append(1)
            featureVector.extend(AbstractPokemon.encoder.encodeItemList(pokemon.item))
        else:
            # If the item is not known we add a zero
            featureVector.extend([0] * (1 + AbstractPokemon.encoder.NUM_UNIQUE_ITEMS))

        # Protect counter
        if pokemon.protect_counter and pokemon in battle.all_active_pokemons:
            featureVector.append(pokemon.protect_counter)
        else:
            featureVector.append(0)

        # Status
        statusToret = [0] * len(AbstractPokemon.encoder.STATUS)

        if (
            pokemon.status
            and pokemon.status.name.lower() in AbstractPokemon.encoder.STATUS
        ):
            statusToret[
                AbstractPokemon.encoder.STATUS.index(pokemon.status.name.lower())
            ] = max(pokemon.status_counter, 1)

        featureVector += statusToret

        # Volatile status effects
        effects = [0] * len(_VOLATILE_STATUS_EFFECTS)

        if pokemon in battle.all_active_pokemons and not pokemon.fainted:
            for effect, counter in pokemon.effects.items():
                if effect.name in AbstractPokemon.encoder.VOLATILE_STATUS:
                    effects[AbstractPokemon.encoder.VOLATILE_STATUS[effect.name]] = max(
                        counter, 1
                    )

        featureVector += effects

        # If the pokemon is doing something that takes 2 turns
        if (
            pokemon.active
            and not pokemon.fainted
            and pokemon in battle.all_active_pokemons
        ):
            # If the pokemon must recharge
            featureVector.append(int(pokemon.must_recharge))

            # If the pokemon is preparing something
            featureVector.append(int(pokemon.preparing))

            # If the pokemon is preparing a move
            if (
                pokemon.preparing_move is not None
                and pokemon.preparing_move in pokemon.moves.values()
            ):
                featureVector.append(
                    (list(pokemon.moves.values()).index(pokemon.preparing_move) + 1) / 4
                )
            else:
                featureVector.append(0)

            # If the pokemon is preparing a target
            featureVector.append(1 if pokemon.preparing_target else 0)
        else:
            featureVector.extend([0] * 4)

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
