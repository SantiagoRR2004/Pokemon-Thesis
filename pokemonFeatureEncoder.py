import serverControl
import json
import os
import re


class PokemonFeatureEncoder:
    """
    This class is to transform
    strings into integer values

    The important functions are:
        - encodeForm(form: str) -> int:
        - encodeAbility(ability: str) -> int:
        - encodeItem(item: str) -> int:
        - encodeMove(move: str) -> int:
    """

    NUM_UNIQUE_FORMS: int
    NUM_UNIQUE_ABILITIES: int
    NUM_UNIQUE_ITEMS: int
    NUM_UNIQUE_MOVES: int

    def __init__(self) -> None:
        serverControl.downloadPokemonShowdown()
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        self.dataPath = os.path.join(currentDirectory, "pokemon-showdown", "data")
        self.preparePokemonForms()
        self.prepareAbilities()
        self.prepareItems()
        self.prepareMoves()

    def removeFunctions(self, text: str) -> str:

        # Pattern to find function declaration start: name(params){
        startPattern = re.compile(r"\w+\s*\([^)]*\)\s*\{")

        i = 0
        result = []
        while i < len(text):
            match = startPattern.search(text, i)
            if not match:
                # No more functions, append rest and break
                result.append(text[i:])
                break

            start, bracePos = match.start(), match.end() - 1

            # Append text before function start
            result.append(text[i:start])

            # Now find the matching closing brace for this function
            braceCount = 1
            j = bracePos + 1
            while j < len(text) and braceCount > 0:
                if text[j] == "{":
                    braceCount += 1
                elif text[j] == "}":
                    braceCount -= 1
                j += 1

            # j is now the position after the matching closing brace

            # Skip any whitespace to check next non-whitespace char
            k = j
            while k < len(text) and text[k].isspace():
                k += 1

            # If the next non-whitespace char is a comma, skip it too
            if k < len(text) and text[k] == ",":
                j = k + 1

            # Skip the function (and possible trailing comma)
            i = j

        return "".join(result)

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

        # Remove multicomment blocks /* */
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

        # Remove comments
        content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE).rstrip()

        # Remove multiple tabulators or spaces
        content = re.sub(r"[ \t]{2,}", " ", content)

        # Remove multiple newlines
        content = re.sub(r"\n{2,}", "\n", content)

        # Strip leading and trailing whitespace
        content = content.strip()

        # Remove the first and last lines
        content = content.splitlines()[1:-1]

        # Rejoin and add dict brackets
        content = "{" + "\n".join(content) + "}"

        # Eliminate JavaScript functions
        content = self.removeFunctions(content)

        # Eliminate: onRestart: () => null,
        content = re.sub(r",\s*onRestart\s*:\s*\(\)\s*=>\s*null\s*", "", content)

        # Add quotes around unquoted keys
        content = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', content)

        # Quote numeric keys like 0: => "0":
        content = re.sub(r"([{,]\s*)(\d+)(\s*:)", r'\1"\2"\3', content)

        # Remove trailing commas before } or ]
        content = re.sub(r",(\s*[}\]])", r"\1", content)

        # Replace single-quoted values with double quotes only when they are around a colon
        content = re.sub(r":\s*\'([^\']*)\'", r': "\1"', content)  # After
        content = re.sub(r"\'([^\']*)\'\s*:", r'"\1":', content)  # Before

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

    def encodeFormList(self, form: str, backup: str) -> list[int]:
        """
        Encodes a Pokemon form and its backup into a list of integers.

        Args:
            - form (str): The primary Pokemon form to encode.
            - backup (str): The backup Pokemon form to encode.

        Returns:
            - list[int]: A list of integers where the index corresponds to the form.
        """
        toret = [0] * self.NUM_UNIQUE_FORMS
        formIndex = self.encodeForm(form)
        if formIndex == -1:
            formIndex = self.encodeForm(backup)
        if formIndex != -1:
            toret[formIndex] = 1
        else:
            raise ValueError(f"Form '{form}' and backup '{backup}' not found.")
        return toret

    def prepareAbilities(self) -> None:
        """
        Prepares the abilities by extracting them from the TypeScript file.

        This method reads the `abilities.ts` file, extracts the abilities data,
        and creates a mapping of abilities to unique integer encodings.

        Args:
            - None

        Returns:
            - None
        """
        abilitiesPath = os.path.join(self.dataPath, "abilities.ts")

        data = self.extractDict(abilitiesPath)

        # Eliminate the ones with negative numbers
        validAbilities = {
            ability: abilityDetails
            for ability, abilityDetails in data.items()
            if abilityDetails["num"] > 0
        }

        self.NUM_UNIQUE_ABILITIES = len(validAbilities)

        self.abilityEncoder = {
            ability: i for i, ability in enumerate(validAbilities.keys())
        }

    def encodeAbility(self, ability: str) -> int:
        """
        Encodes a Pokemon ability into an integer.

        Args:
            ability (str): The Pokemon ability to encode.

        Returns:
            int: The encoded integer value of the ability.
            If the ability is not found, returns -1.
        """
        return self.abilityEncoder.get(ability, -1)

    def encodeAbilityList(self, abilities: list[str]) -> list[int]:
        """
        Encodes a Pokemon ability into a list of integers.

        If there are multiple abilities, each is given the probability of
        1 divided by the number of abilities.

        Args:
            - abilities (str): A list of Pokemon abilities to encode.

        Returns:
            - list[int]: A list of integers where the index corresponds to the ability.
        """
        toret = [0] * self.NUM_UNIQUE_ABILITIES
        for ability in abilities:
            abilityIndex = self.encodeAbility(ability)
            if abilityIndex != -1:
                toret[abilityIndex] = 1 / len(abilities)
            else:
                raise ValueError(f"Ability '{ability}' not found.")
        return toret

    def prepareItems(self) -> None:
        """
        Prepares the items by extracting them from the TypeScript file.

        This method reads the `items.ts` file, extracts the items data,
        and creates a mapping of items to unique integer encodings.

        Args:
            - None

        Returns:
            - None
        """
        itemsPath = os.path.join(self.dataPath, "items.ts")

        data = self.extractDict(itemsPath)

        # Eliminate the ones with negative numbers
        validItems = {
            item: itemDetails
            for item, itemDetails in data.items()
            if itemDetails["num"] > 0
        }

        self.NUM_UNIQUE_ITEMS = len(validItems)

        self.itemEncoder = {item: i for i, item in enumerate(validItems.keys())}

    def encodeItem(self, item: str) -> int:
        """
        Encodes a Pokemon item into an integer.

        Args:
            item (str): The Pokemon item to encode.

        Returns:
            int: The encoded integer value of the item.
            If the item is not found, returns -1.
        """
        return self.itemEncoder.get(item, -1)

    def encodeItemList(self, item: str) -> list[int]:
        """
        Encodes a Pokemon item into a list of integers.

        Args:
            - item (str): The Pokemon item to encode.

        Returns:
            - list[int]: A list of integers where the index corresponds to the item.
        """
        toret = [0] * self.NUM_UNIQUE_ITEMS
        itemIndex = self.encodeItem(item)
        if itemIndex != -1:
            toret[itemIndex] = 1
        else:
            raise ValueError(f"Item '{item}' not found.")
        return toret

    def prepareMoves(self) -> None:
        """
        Prepares the moves by extracting them from the TypeScript file.

        This method reads the `moves.ts` file, extracts the moves data,
        and creates a mapping of moves to unique integer encodings.

        Args:
            - None

        Returns:
            - None
        """
        movesPath = os.path.join(self.dataPath, "moves.ts")

        data = self.extractDict(movesPath)

        # Eliminate the ones with negative numbers
        validMoves = {
            move: moveDetails
            for move, moveDetails in data.items()
            if moveDetails["num"] > 0
        }

        self.NUM_UNIQUE_MOVES = len(validMoves)

        self.moveEncoder = {move: i for i, move in enumerate(validMoves.keys())}

    def encodeMove(self, move: str) -> int:
        """
        Encodes a Pokemon move into an integer.

        Args:
            move (str): The Pokemon move to encode.

        Returns:
            int: The encoded integer value of the move.
            If the move is not found, returns -1.
        """
        return self.moveEncoder.get(move, -1)

    def encodeMoveList(self, move: str) -> list[int]:
        """
        Encodes a Pokemon move into a list of integers.

        Args:
            - move (str): The Pokemon move to encode.

        Returns:
            - list[int]: A list of integers where the index corresponds to the move.
        """
        toret = [0] * self.NUM_UNIQUE_MOVES
        moveIndex = self.encodeMove(move)
        if moveIndex != -1:
            toret[moveIndex] = 1
        else:
            raise ValueError(f"Move '{move}' not found.")
        return toret


if __name__ == "__main__":
    encoder = PokemonFeatureEncoder()
    print(f"Number of unique forms: {encoder.NUM_UNIQUE_FORMS}")
    print(f"Number of unique abilities: {encoder.NUM_UNIQUE_ABILITIES}")
    print(f"Number of unique items: {encoder.NUM_UNIQUE_ITEMS}")
    print(f"Number of unique moves: {encoder.NUM_UNIQUE_MOVES}")
