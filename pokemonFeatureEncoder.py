import json
import os
import re


class PokemonFeatureEncoder:
    """
    This class is to transform
    strings into integer values
    """

    def __init__(self) -> None:
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        pokedexPath = os.path.join(
            currentDirectory, "pokemon-showdown", "data", "pokedex.ts"
        )
        data = self.extractDict(pokedexPath)

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


if __name__ == "__main__":
    encoder = PokemonFeatureEncoder()
