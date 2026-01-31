from pokemonFeatureEncoder import PokemonFeatureEncoder
from poke_env.battle import Move
import inspect
import moves

encoder = PokemonFeatureEncoder()


# Get all classes defined in the module
classes = inspect.getmembers(moves, inspect.isclass)


for name, obj in classes:
    if name != "AbstractMove":
        for moveName in encoder.moveEncoder:
            length = len(obj.getFeatures(Move(moveName, 9)))
            assert (
                obj.N_F_MOVE == length
            ), f"Move {moveName} for class {name} {obj.N_F_MOVE} != {length}."

        print(f"Class {name} is valid with {obj.N_F_MOVE} features.")
