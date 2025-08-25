from pokemons import AbstractPokemon


class Pokemon05(AbstractPokemon):

    N_F_POKEMON = (
        1
        + 1
        + 1
        + 1
        + AbstractPokemon.encoder.N_F_TYPES
        + AbstractPokemon.encoder.N_F_TYPES
        + (1 + 1)
        + (1 + 1)
        + 1
        + 1
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
            - The ability's presence indicator
            - The ability of the pokemon (encoded as an integer)
            - The item's presence indicator
            - The item of the pokemon (encoded as an integer)
            - If it is the first turn of the pokemon (boolean)
            - The HP fraction of the pokemon
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

        # Add the ability
        ability = self.encoder.encodeAbility(pokemon.ability)
        if ability == -1:
            # We don't know the ability
            featureVector.extend([0, 0])
        else:
            featureVector.extend([1, ability])

        # Add the item
        item = self.encoder.encodeItem(pokemon.item)
        if item == -1:
            # We don't know the item
            featureVector.extend([0, 0])
        else:
            featureVector.extend([1, item])

        # If it is the first turn of the pokemon
        featureVector.append(int(pokemon.first_turn))

        # The HP fraction of the pokemon
        featureVector.append(pokemon.current_hp_fraction)

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
