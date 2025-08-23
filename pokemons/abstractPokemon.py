from poke_env.battle.effect import _VOLATILE_STATUS_EFFECTS
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

    N_F_TYPES = 20  # Tera stellar and Pawmot

    STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
    BOOSTABLE_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    STATUS = ["brn", "frz", "par", "psn", "tox", "slp"]  # We skip "fnt"
    VOLATILE_STATUS = {}
    for index, status in enumerate(
        sorted(_VOLATILE_STATUS_EFFECTS, key=lambda s: s.name)
    ):
        name = status.name
        VOLATILE_STATUS[name] = index
        VOLATILE_STATUS[name.replace("_", "").lower()] = index

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
    def getFeatures(self, pokemon: Pokemon) -> list[float]:
        """
        This method will return the features of the Pokemon as a list of floats.

        Args:
            - pokemon (Pokemon): The Pokemon to encode.

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
