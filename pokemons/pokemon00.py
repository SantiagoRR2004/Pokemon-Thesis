from pokemons import AbstractPokemon


class Pokemon00(AbstractPokemon):

    N_F_POKEMON = 1

    def getFeatures(self, pokemon, battle) -> list[float]:
        """
        This method will encode a pokemon into a feature vector

        The feature vector will have:
            - The form of the pokemon (integer)
            - The 4 moves of the pokemon:
                - The encoded move

        Args:
            - pokemon (Pokemon): The pokemon to be encoded

        Returns:
            - list[float]: The feature vector of the pokemon
        """
        featureVector = []

        # The form of the pokemon
        form = self.encoder.encodeForm(pokemon.species)
        if form == -1:
            # It was a cosmetic form
            form = self.encoder.encodeForm(pokemon.base_species)
        featureVector.append(form)

        # The moves
        moves = []
        for move in pokemon.moves.values():
            # We encode the move
            moves.extend(self.moveEncoder.getFeatures(move))
        # We fill with zeros if the pokemon has less than 4 moves
        moves += (
            [0]
            * self.moveEncoder.getNumberOfFeatures()
            * (4 - len(pokemon.moves.values()))
        )
        featureVector += moves

        return featureVector
