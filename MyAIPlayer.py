from poke_env.environment import AbstractBattle, Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
import pokemonFeatureEncoder
import neat
import numpy as np


class AIPlayer(Player):
    """
    This will only be made
    to work with gen9randombattle
    """

    encoder = pokemonFeatureEncoder.PokemonFeatureEncoder()

    N_F_TYPES = 20  # Tera stellar and Pawmot
    N_F_POKEMON = 1 + N_F_TYPES + 1 + 1 + 1 + 4
    N_F_TOTAL = (1 + N_F_POKEMON) * 12

    N_OUTPUTS = 14  # 8 moves + 6 switches

    def __init__(self, *args, network: neat.nn.FeedForwardNetwork, **kwargs) -> None:
        """
        This is the constructor of the class

        Args:
            *args:
            - network (neat.nn.FeedForwardNetwork): The neural network that will be used to make decisions
            **kwargs:

        Returns:
            - None
        """

        super().__init__(*args, **kwargs)
        self.neuralNetwork = network

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        This is the most important method because it must be implemented

        Args:
            battle (AbstractBattle): The current battle

        Returns:
            BattleOrder: The move to be executed
        """

        inputs = self.getInputs(battle)

        outputs = self.neuralNetwork.activate(inputs)

        return self.translateOutputs(outputs, battle)

    def getInputs(self, battle: AbstractBattle) -> list[float]:
        """
        This method will return the inputs for the neural network

        For each pokemon we have a presence indicator
        and then the encoding of the pokemon.

        Args:
            battle (AbstractBattle): The current battle

        Returns:
            list: The inputs for the neural network
        """
        inputs = []

        # First our team
        for pokemon in battle.team.values():
            inputs += [1] + self.encodePokemon(pokemon)
        # Fill with zeros if unknown pokemon
        inputs += (6 - len(battle.team)) * ([0] + [0.0] * self.N_F_POKEMON)

        # The opponent's team
        for pokemon in battle.opponent_team.values():
            inputs += [1] + self.encodePokemon(pokemon)
        # Fill with zeros if unknown pokemon
        inputs += (6 - len(battle.opponent_team)) * ([0] + [0.0] * self.N_F_POKEMON)

        return inputs

    def encodePokemon(self, pokemon: Pokemon) -> list[float]:
        """
        This method will encode a pokemon into a feature vector

        The feature vector will have:
            - The form of the pokemon (encoded as an integer)
            - The original types of the pokemon (encoded as a list of {self.N_F_TYPES} integers)
            - The ability of the pokemon (encoded as an integer)
            - The item of the pokemon (encoded as an integer)
            - The HP fraction of the pokemon
            - The 4 moves of the pokemon:
                - The name of the move (encoded as an integer)

        Args:
            - pokemon (Pokemon): The pokemon to be encoded

        Returns:
            - list[float]: The feature vector of the pokemon
        """
        featureVector = []

        # The form of the pokemon
        form = self.encoder.encodeForm(pokemon.species)
        if form == -1:
            # It was a cosmetic form
            form = self.encoder.encodeForm(pokemon.base_species)
        featureVector.append(form)

        # The original type of the pokemon
        types = [0] * self.N_F_TYPES
        for t in pokemon.original_types:
            types[t.value - 1] = 1
        featureVector += types

        # Add the ability
        featureVector.append(self.encoder.encodeAbility(pokemon.ability))

        # Add the item
        featureVector.append(self.encoder.encodeItem(pokemon.item))

        # The HP fraction of the pokemon
        featureVector.append(pokemon.current_hp_fraction)

        # The moves
        moves = []
        for move in pokemon.moves.values():
            # We encode the move
            moves.append(self.encoder.encodeMove(move.id))
        # We fill with zeros if the pokemon has less than 4 moves
        moves += [0] * (4 - len(moves))
        featureVector += moves

        return featureVector

    def encodePokemonComplex(self, pokemon: Pokemon) -> list[float]:
        """
        This stores the thing that are not yet in encodePokemon

        The feature vector will have:
            - The current types of the pokemon (encoded as a list of {self.N_F_TYPES} integers)
            - If the pokemon has already tera'd (encoded as an integer)
            - If the tera type is known (encoded as an integer)
            - The tera type of the pokemon (encoded as a list of {self.N_F_TYPES} integers)
        """
        featureVector = []

        # The current pokemon type
        cTypes = [0] * self.N_F_TYPES
        for t in pokemon.types:
            cTypes[t.value - 1] = 1
        featureVector += cTypes

        # If the pokemon has already tera'd
        featureVector.append(int(pokemon.is_terastallized))

        # If the tera type is known
        if pokemon.tera_type:
            tTypes = [0] * self.N_F_TYPES
            tTypes[pokemon.tera_type.value - 1] = 1
            featureVector += [1] + tTypes
        else:
            # If the tera type is not known we add a zero
            featureVector += [0] + ([0] * self.N_F_TYPES)

        return featureVector

    def translateOutputs(
        self, outputs: list[float], battle: AbstractBattle
    ) -> BattleOrder:
        """
        This method will translate the outputs of the neural network into a BattleOrder

        There should be 14 outputs:
            - 8 for the moves of the current pokemon
                - Alternating between not teraing and teraing
            - 6 for the switches of the team

        Args:
            outputs (list[float]): The outputs of the neural network

        Returns:
            BattleOrder: The move to be executed
        """
        # All the moves plus all the switches
        validOrders = []
        for move in battle.available_moves:
            validOrders.append(BattleOrder(move))
            if battle.can_tera:
                validOrders.append(BattleOrder(move, terastallize=True))
        validOrders += [BattleOrder(switch) for switch in battle.available_switches]

        # If there a no posssible we return the default move
        if not validOrders:
            return self.choose_default_move()

        currentPokemon = battle.active_pokemon
        allMoves = list(currentPokemon.moves.values())

        validMoves: list[bool] = [m in battle.available_moves for m in allMoves]

        # We check if the pokemon can tera
        if battle.can_tera:
            # Duplicate each valid moves to include the tera option
            validMoves = [m for m in validMoves for _ in range(2)]
        else:
            # Add a False
            validMoves = [item for m in validMoves for item in (m, False)]

        # The full team
        fullTeam = list(battle.team.values())

        # The order of the team is always the same
        validSwitches: list[bool] = [p in battle.available_switches for p in fullTeam]

        # We eliminate all the impossible outputs
        validOutputs = np.array(
            [o for o, flag in zip(outputs, validMoves + validSwitches) if flag]
        )

        # We haven't found any valid outputs
        if validOutputs.size == 0:
            # This probably means the pokemon must struggle and can't switch
            return self.choose_default_move()

        # We normalize the outputs
        total = np.sum(validOutputs)
        if total == 0:
            # Softmax would be slower and there are no negative values
            # The minimum would be zero
            validOutputs = np.ones_like(validOutputs) / len(validOutputs)
        else:
            validOutputs /= total

        # We use the probabilities to choose the move
        chosenIndex = np.random.choice(len(validOutputs), p=validOutputs)

        # We return the chosen order
        return validOrders[chosenIndex]
