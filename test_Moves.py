from pokemonFeatureEncoder import PokemonFeatureEncoder
from players import AIPlayerS
from poke_env.battle import Move


encoder = PokemonFeatureEncoder()

player = AIPlayerS(network="BlaBlaBla")

for moveName in encoder.moveEncoder:
    player.encodeMove(Move(moveName, 9))
