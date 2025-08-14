from poke_env.battle import Pokemon
from poke_env.battle import AbstractBattle
from players.AbstractAIPlayer import AbstractAIPlayer


class AIPlayer(AbstractAIPlayer):
    """
    This will only be made
    to work with gen9randombattle
    """

    N_F_POKEMON = 1 + 2 + 2 + 1 + 4
    N_F_BATTLE = 2 + 12

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
        inputs += (
            [0]
            * (6 - len(battle.team))
            * (
                1
                + (
                    self.N_F_POKEMON
                    + 4 * self.moveFeatureExtractor.getNumberOfFeatures()
                )
            )
        )

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
        inputs += (
            [0]
            * (6 - len(battle.opponent_team))
            * (
                1
                + (
                    self.N_F_POKEMON
                    + 4 * self.moveFeatureExtractor.getNumberOfFeatures()
                )
            )
        )

        return inputs

    def encodePokemon(self, pokemon: Pokemon) -> list[float]:
        """
        This method will encode a pokemon into a feature vector

        The feature vector will have:
            - The form of the pokemon (encoded as an integer)
            - The ability's presence indicator
            - The ability of the pokemon (encoded as an integer)
            - The item's presence indicator
            - The item of the pokemon (encoded as an integer)
            - The HP fraction of the pokemon
            - The 4 moves of the pokemon:
                - The presence indicator of the move
                - The name of the move (encoded as an integer)

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

        # The HP fraction of the pokemon
        featureVector.append(pokemon.current_hp_fraction)

        # The moves
        moves = []
        for move in pokemon.moves.values():
            # We encode the move
            moves.append(1)
            moves.extend(self.moveFeatureExtractor.getFeatures(move))
        # We fill with zeros if the pokemon has less than 4 moves
        moves += (
            [0]
            * (4 - len(pokemon.moves.values()))
            * (1 + self.moveFeatureExtractor.getNumberOfFeatures())
        )

        featureVector += moves

        return featureVector
