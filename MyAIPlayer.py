from poke_env.environment import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
import neat


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

        return self.translateOutputs(outputs)

    def getInputs(battle: AbstractBattle) -> list[float]:
        """
        This method will return the inputs for the neural network

        Args:
            battle (AbstractBattle): The current battle

        Returns:
            list: The inputs for the neural network
        """
        pass

    def translateOutputs(outputs: list[float]) -> BattleOrder:
        """
        This method will translate the outputs of the neural network into a BattleOrder

        Args:
            outputs (list[float]): The outputs of the neural network

        Returns:
            BattleOrder: The move to be executed
        """
        pass
