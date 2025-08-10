from poke_env.battle import AbstractBattle, Pokemon, Move
from players.AbstractAIPlayer import AbstractAIPlayer
import numpy as np


class AIPlayerS(AbstractAIPlayer):
    """
    This will have all the information
    """

    N_F_TYPES = 20  # Tera stellar and Pawmot
    N_F_MOVE = (
        AbstractAIPlayer.encoder.NUM_UNIQUE_MOVES
        + N_F_TYPES
        + 3
        + 3
        + 1
        + 1
        + 1
        + 2
        + 1
        + 1
        + 3
        + 7
        + (1 + 1 + 1)
        + (1 + 1 + 1 + 1)
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
    )
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
            - The name of the move (encoder.NUM_UNIQUE_MOVES One-Hot Encoding)
            - The type of the move ({self.N_F_TYPES} One-Hot Encoding)
            - The category of the move (3 One-Hot Encoding)
                - PHYSICAL
                - SPECIAL
                - STATUS
            - The defense the move targets (3 One-Hot Encoding):
                - PHYSICAL
                - SPECIAL
                - STATUS
            - The base power of the move (normalized to a float between 0 and 1)
                Divided by 120 because higher than that and they become very rare.
            - If the base power is preset or level dependent (boolean)
            - The accuracy of the move (a float between 0 and 1)
                It is already given between 0 and 1.
            - The remaining PP percentage of the move
            - The maximum PP of the move (a float between 0 and 1)
                We use the base PP without using PP up
            - The priority of the move (float between -1 to 1)
            - The critical hit ratio (float between 0 and 1)
            - Number of hits (3 floats between 0 and 1):
                - The minimum number of hits
                - The expected number of hits
                - The maximum number of hits
            - The boosts the move gives (7 floats between -1 and 1):
                - Attack
                - Defense
                - Special Attack
                - Special Defense
                - Speed
                - Accuracy
                - Evasion
            - The protection data (3 booleans):
                - If the move is a protect move (1 if true, 0 otherwise)
                - If the move increases the protect counter (1 if true, 0 otherwise)
                - If it is a side protect move (1 if true, 0 otherwise)
            - If the move creates something on the battlefield (4 booleans):
                - If the move is a weather move (1 if true, 0 otherwise)
                - If the move is a pseudo weather move (1 if true, 0 otherwise)
                - If the move creates a side condition (1 if true, 0 otherwise)
                - If the move creates a terrain (1 if true, 0 otherwise)

        The following features are not used:
            - Anything related to z-moves
            - Anything related to dynamax
            - move.is_empty (Seems to always be False)
            - move.non_ghost_target (It is only used for curse)
            - move.no_pp_boosts (It is only used for revival blessing)
            - move.sleep_usable (It is only used for Sleep Talk and Snore)
            - move.steals_boosts (It is only used for Spectral Thief)
            - move.request_target (Seems to be not used)
            - move.use_target_offensive (It is only used for Foul Play)
            - move.entry (Everything here is somewhere else)
            - move.target

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

        # Add the defense the move targets
        defensiveCategory = [0] * 3
        defensiveCategory[move.defensive_category.value - 1] = 1
        toret.extend(defensiveCategory)

        # Add the base power of the move
        toret.append(move.base_power / 120)

        # Add if the base power is preset or level dependent
        toret.append(1 if move.damage else 0)

        # Add the accuracy of the move
        toret.append(move.accuracy)

        # Add the PP of the move
        toret.append(move.current_pp / move.max_pp)
        toret.append(move.max_pp * 0.625 / 40)

        # Add the priority of the move
        toret.append(move.priority / 8)

        # Add the critical hit ratio
        toret.append(move.crit_ratio / 6)

        # The number of hits
        toret.append(move.n_hit[0] / 5)
        toret.append(move.expected_hits / 5)
        toret.append(move.n_hit[1] / 5)

        # The boosts the move gives or takes
        if move.boosts or move.self_boost:
            boostsCombined = {**(move.boosts or {}), **(move.self_boost or {})}
            for stat in (set(self.STATS) - {"hp"}).union({"accuracy", "evasion"}):
                toret.append(boostsCombined.get(stat, 0) / 2)
        else:
            toret += [0] * 7

        ## The protection data
        # If the move is a protect move
        toret.append(int(move.is_protect_move))

        # If the move increases the protect counter
        toret.append(int(move.is_protect_counter))

        # If it is a side protect move
        toret.append(int(move.is_side_protect_move))

        ## The move creates something on the battlefield
        # If the move is a weather move
        toret.append(1 if move.weather else 0)

        # If the move is a pseudo weather move
        toret.append(1 if move.pseudo_weather else 0)

        # If the move creates a side condition
        toret.append(1 if move.side_condition else 0)

        # If the move creates a terrain
        toret.append(1 if move.terrain else 0)

        # If the move can break trough protect
        toret.append(int(move.breaks_protect))

        # The percentage the move drains
        toret.append(move.drain)

        # The percentage the move heals
        toret.append(move.heal)

        # The move forces the opponent to switch
        toret.append(int(move.force_switch))

        # If the move ignores abilities
        toret.append(int(move.ignore_ability))

        # If the move ignores defensive boosts
        toret.append(int(move.ignore_defensive))

        # If the move ignores evasion
        toret.append(int(move.ignore_evasion))

        # If the move ignores immunities
        toret.append(int(move.ignore_immunity))

        # The percentage of recoil damage
        toret.append(move.recoil)

        # If the move is a self-destruct move
        toret.append(1 if move.self_destruct else 0)

        # If the move is a self-switch move
        toret.append(1 if move.self_switch else 0)

        # If the move adds a slot condition
        toret.append(1 if move.slot_condition else 0)

        # If the move is a stalling move
        toret.append(int(move.stalling_move))

        # If the move tries to give a status condition
        toret.append(1 if move.status else 0)

        # If the move thaws
        toret.append(int(move.thaws_target))

        # If the move tries to inflict a volatile status
        toret.append(1 if move.volatile_status else 0)

        if move.flags:
            # For later use
            pass

        if move.secondary:
            # For later use
            pass

        if move.deduced_target or move.target:
            pass  # For later use

        return toret


if __name__ == "__main__":
    print(f"Number of features: {AIPlayerS.N_F_TOTAL}")
