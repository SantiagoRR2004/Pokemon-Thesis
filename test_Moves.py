from pokemonFeatureEncoder import PokemonFeatureEncoder
from moves import MoveS
from poke_env.battle import Move


encoder = PokemonFeatureEncoder()

for moveName in encoder.moveEncoder:
    MoveS.getFeatures(Move(moveName, 9))
