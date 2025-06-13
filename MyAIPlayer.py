from poke_env.environment import AbstractBattle, Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
import pokemonFeatureEncoder
import torch.nn as nn
import numpy as np
import torch


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

    def __init__(self, *args, network: nn.Module, **kwargs) -> None:
        """
        This is the constructor of the class

        Args:
            *args:
            - network (nn.Module): The neural network that will be used to make decisions
            **kwargs:

        Returns:
            - None
        """

        super().__init__(*args, **kwargs)
        self.neuralNetwork = network
        self.log_probs = []

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        This is the most important method because it must be implemented

        Args:
            battle (AbstractBattle): The current battle

        Returns:
            BattleOrder: The move to be executed
        """

        inputs = torch.tensor(self.getInputs(battle))

        # Raw network outputs
        logits = self.neuralNetwork(inputs)

        mask, moves = self.translateOutputs(battle)

        if not any(mask):
            log_prob = torch.tensor(0.0)
            self.log_probs.append(log_prob)
            return self.choose_default_move()

        # We apply the mask to the logits
        masked_logits = logits.clone()
        masked_logits[~mask] = float("-inf")

        # We apply softmax to get probabilities
        probs = torch.softmax(masked_logits, dim=0)

        # We sample an action from the distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_probs.append(log_prob)

        return moves[action.item()]

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
        self, battle: AbstractBattle
    ) -> tuple[torch.Tensor, list[BattleOrder]]:
        """
        This method will return the moves and switches that can be made

        There should be 14 outputs:
            - 8 for the moves of the current pokemon
                - Alternating between not teraing and teraing
            - 6 for the switches of the team

        Args:
            - battle (AbstractBattle): The current battle

        Returns:
            - tuple[torch.Tensor, list[BattleOrder]]:
                - A tensor of booleans indicating which moves and switches are valid
                - A list of BattleOrder objects corresponding to the moves and switches
        """
        # All the moves plus all the switches
        allOrders = []
        validOrders = []

        # First we add the moves of the active pokemon
        for move in battle.active_pokemon.moves.values():
            allOrders.append(BattleOrder(move))
            validOrders.append(move in battle.available_moves)

            allOrders.append(BattleOrder(move, terastallize=True))
            validOrders.append(move in battle.available_moves and bool(battle.can_tera))

        # Then we add the switches
        for pokemon in battle.team.values():
            allOrders.append(BattleOrder(pokemon))
            validOrders.append(pokemon in battle.available_switches)

        return torch.tensor(validOrders, dtype=torch.bool), allOrders
