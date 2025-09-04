from poke_env.battle import AbstractBattle
from players import AbstractAIPlayer


class AIPlayer00(AbstractAIPlayer):

    N_F_BATTLE = 0

    def getInputs(self, battle: AbstractBattle) -> list[float]:
        """
        This method will return the inputs for the neural network

        The feature vector will have:
            - The player's team ({self.N_F_POKEMON} features per pokemon):
                - The encoding of the pokemon
            - The opponent's team ({self.N_F_POKEMON} features per pokemon):
                - The encoding of the pokemon

        Args:
            - battle (AbstractBattle): The current battle

        Returns:
            - list: The inputs for the neural network
        """
        inputs = []

        # First our team
        for pokemon in battle.team.values():
            inputs += self.pokemonFeatureExtractor.getFeatures(pokemon, battle)
        # Fill with zeros if unknown pokemon
        inputs += (
            [0]
            * (6 - len(battle.team))
            * self.pokemonFeatureExtractor.getNumberOfFeatures()
        )

        # The opponent's team
        for pokemon in battle.opponent_team.values():
            inputs += self.pokemonFeatureExtractor.getFeatures(pokemon, battle)
        # Fill with zeros if unknown pokemon
        inputs += (
            [0]
            * (6 - len(battle.opponent_team))
            * self.pokemonFeatureExtractor.getNumberOfFeatures()
        )

        return inputs
