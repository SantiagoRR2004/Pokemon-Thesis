from poke_env.battle.effect import _VOLATILE_STATUS_EFFECTS
from pokemons import AbstractPokemon


class PokemonS(AbstractPokemon):

    N_F_POKEMON = (
        1
        + 1
        + 1
        + AbstractPokemon.encoder.NUM_UNIQUE_FORMS
        + AbstractPokemon.N_F_TYPES
        + AbstractPokemon.N_F_TYPES
        + 1
        + 1
        + AbstractPokemon.N_F_TYPES
        + 1
        + 6
        + 1
        + 3
        + (1 + AbstractPokemon.encoder.NUM_UNIQUE_ABILITIES)
        + 1
        + 1
        + 1
        + (2 * 6)
        + 7
        + (1 + AbstractPokemon.encoder.NUM_UNIQUE_ITEMS)
        + 1
        + (1 + 1 + 1 + 1)
        + len(AbstractPokemon.STATUS)
        + len(_VOLATILE_STATUS_EFFECTS)
        + 4
    )
