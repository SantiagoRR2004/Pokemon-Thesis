from poke_env.environment import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
import neat
import numpy as np


class AIPlayer(Player):
    """
    This will only be made
    to work with gen9randombattle
    """

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

    enablePrint = False
    separator = "-" * 80

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

        Args:
            battle (AbstractBattle): The current battle

        Returns:
            list: The inputs for the neural network
        """
        return [np.random.random()]

    def translateOutputs(
        self, outputs: list[float], battle: AbstractBattle
    ) -> BattleOrder:
        """
        This method will translate the outputs of the neural network into a BattleOrder

        For now we are going to have 10 outputs. 4 will be for the move and 6
        to switch to a different pokemon. We don't take into account tera
        for now.

        Args:
            outputs (list[float]): The outputs of the neural network

        Returns:
            BattleOrder: The move to be executed
        """
        # All the moves plus all the switches
        validOrders = [BattleOrder(move) for move in battle.available_moves] + [
            BattleOrder(switch) for switch in battle.available_switches
        ]

        return np.random.choice(validOrders)

        currentPokemon = battle.active_pokemon
        allMoves = list(currentPokemon.moves.values())

        validMoves: list[bool] = [m in battle.available_moves for m in allMoves]

        # The full team
        fullTeam = list(battle.team.values())

        # The order of the team is always the same
        validSwitches: list[bool] = [p in battle.available_switches for p in fullTeam]

        # We eliminate all the impossible outputs
        validOutputs = np.array(
            [o for o, flag in zip(outputs, validMoves + validSwitches) if flag]
        )

        # We normalize the outputs
        validOutputs /= np.sum(validOutputs)
        ## CHECK IF SOFTMAX IS BETTER

        # We use the probabilities to choose the move
        chosenIndex = np.random.choice(len(validOutputs), p=validOutputs)

        # We return the chosen order
        return validOrders[chosenIndex]
