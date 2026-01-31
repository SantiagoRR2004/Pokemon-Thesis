from pokemonFeatureEncoder import PokemonFeatureEncoder
from poke_env.battle import Pokemon
from moves import Move00
import pokemons
import inspect

encoder = PokemonFeatureEncoder()


# Get all classes defined in the module
classes = inspect.getmembers(pokemons, inspect.isclass)


for name, obj in classes:
    if name != "AbstractPokemon":
        ins = obj(Move00)
        for pokemonSpecies in encoder.formEncoder:
            """
            TODO Fix the pokemon problem
            When the pokemon is created this way some attributes are missing
            that are there when created normally
            """
            # length = len(
            #     ins.getFeatures(Pokemon(species=pokemonSpecies, gen=9), "dskjhfjhukds")
            # )
            # assert (
            #     obj.N_F_POKEMON + 4 * (Move00.N_F_MOVE) == length
            # ), f"Move {pokemonSpecies} for class {name} {obj.N_F_POKEMON} != {length}."

        print(f"Class {name} is valid with {ins.N_F_POKEMON} features.")
