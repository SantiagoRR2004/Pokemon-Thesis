import json
import os
import re


class PokemonFeatureEncoder:
    """
    This class is to transform
    strings into integer values

    The important functions are:
        - encodeForm(form: str) -> int:
    """

    def __init__(self) -> None:
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        self.dataPath = os.path.join(currentDirectory, "pokemon-showdown", "data")
        self.preparePokemonForms()

    def extractDict(self, path: str) -> dict:
        """
        Extracts a dictionary from a typeScript file.

        Args:
            path (str): The path to the TypeScript file.

        Returns:
            dict: The extracted dictionary.
        """
        with open(path) as f:
            content = f.read()

        # Remove the first and last lines
        content = content.splitlines()[1:-1]

        # Rejoin and add dict brackets
        content = "{" + "\n".join(content) + "}"

        # Remove comments
        content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE).rstrip()

        # Add quotes around unquoted keys
        content = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', content)

        # Quote numeric keys like 0: => "0":
        content = re.sub(r"([{,]\s*)(\d+)(\s*:)", r'\1"\2"\3', content)

        # Remove trailing commas before } or ]
        content = re.sub(r",(\s*[}\]])", r"\1", content)

        # Replace single-quoted values with double quotes only when they are after a colon
        content = re.sub(r":\s*\'([^\']*)\'", r': "\1"', content)

        return json.loads(content)

    def preparePokemonForms(self) -> None:
        """
        Prepares the Pokemon forms by extracting them from the TypeScript file.

        This method reads the `pokedex.ts` file, extracts the Pokemon data,
        and creates a mapping of Pokemon forms to unique integer encodings.

        Args:
            - None

        Returns:
            - None
        """
        pokedexPath = os.path.join(self.dataPath, "pokedex.ts")

        data = self.extractDict(pokedexPath)

        # Eliminate the ones with negative numbers
        validPokemon = {
            pokemon: pokemonDetails
            for pokemon, pokemonDetails in data.items()
            if pokemonDetails["num"] > 0
        }

        self.NUM_UNIQUE_FORMS = len(validPokemon)

        self.formEncoder = {pokemon: i for i, pokemon in enumerate(validPokemon.keys())}

    def encodeForm(self, form: str) -> int:
        """
        Encodes a Pokemon form into an integer.

        Args:
            form (str): The Pokemon form to encode.

        Returns:
            int: The encoded integer value of the form.
            If the form is not found, returns -1.
        """
        return self.formEncoder.get(form, -1)


if __name__ == "__main__":
    encoder = PokemonFeatureEncoder()
