from poke_env.battle import AbstractBattle
from abc import ABC, abstractmethod
from poke_env.battle import Pokemon
from moves import AbstractMove
import pokemonFeatureEncoder


class AbstractPokemon(ABC):
    """
    This is the abstract class that will be used to represent a Pokemon
    as a feature vector for the neural network.
    """

    encoder = pokemonFeatureEncoder.PokemonFeatureEncoder()

    def __init__(self, moveEncoder: AbstractMove) -> None:
        """
        This constructor requires a move encoder to be passed in.
        This will be used by the child classes to encode the moves of the Pokemon.

        Args:
            - moveEncoder (AbstractMove): The move encoder to use.

        Returns:
            - None
        """
        self.moveEncoder: AbstractMove = moveEncoder

    @abstractmethod
    def getFeatures(self, pokemon: Pokemon, battle: AbstractBattle) -> list[float]:
        """
        This method will return the features of the Pokemon as a list of floats.

        Args:
            - pokemon (Pokemon): The Pokemon to encode.
            - battle (AbstractBattle): The battle context (Used to check active Pokemon)

        Returns:
            - list[float]: The features of the Pokemon.
        """
        pass

    N_F_POKEMON = 0

    def getNumberOfFeatures(self) -> int:
        """
        This method will return the number of features of the Pokemon
        """
        if self.N_F_POKEMON == 0:
            raise ValueError("The number of features has not been set yet")
        return self.N_F_POKEMON + 4 * self.moveEncoder.getNumberOfFeatures()
