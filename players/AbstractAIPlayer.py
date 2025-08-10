from poke_env.battle import AbstractBattle
from poke_env.player.battle_order import SingleBattleOrder
from poke_env.player.player import Player
from abc import ABC, abstractmethod
import pokemonFeatureEncoder
import torch.nn as nn
import torch


class AbstractAIPlayer(Player, ABC):
    """
    This will only be made
    to work with gen9randombattle
    """

    encoder = pokemonFeatureEncoder.PokemonFeatureEncoder()

    N_F_TYPES = 20  # Tera stellar and Pawmot
    N_F_POKEMON = 0
    N_F_TOTAL = 0

    N_OUTPUTS = 14  # 8 moves + 6 switches

    def __init__(
        self, *args, network: nn.Module, critic: nn.Module = None, **kwargs
    ) -> None:
        """
        This is the constructor of the class

        Args:
            *args:
            - network (nn.Module): The neural network that will be used to make decisions
            - critic (nn.Module): The critic network for value estimation (optional)
            **kwargs:

        Returns:
            - None
        """

        super().__init__(*args, **kwargs)
        self.neuralNetwork = network
        self.criticNetwork = critic
        self.reset()

    STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
    BOOSTABLE_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    STATUS = ["brn", "frz", "par", "psn", "tox", "slp"]  # We skip "fnt"
    VOLATILE_STATUS = [
        "flinch",
        "confusion",
        "healblock",
        "salt_cure",
        "saltcure",
        "sparkling_aria",
        "sparklingaria",
        "protect",
        "substitute",
        "encore",
        "locked_move",
        "taunt",
        "roost",
        "glaive_rush",
        "heal_block",
        "curse",
        "must_recharge",
        "yawn",
        "leech_seed",
        "no_retreat",
        "magnet_rise",
        "partially_trapped",
        "destiny_bond",
        "disable",
    ]

    def reset(self) -> None:
        """
        Reset the internal lists for training

        Args:
            - None

        Returns:
            - None
        """
        self.log_probs = {}
        self.values = {}

    def choose_move(self, battle: AbstractBattle) -> SingleBattleOrder:
        """
        This is the most important method because it must be implemented

        Args:
            battle (AbstractBattle): The current battle

        Returns:
            SingleBattleOrder: The move to be executed
        """

        inputs = torch.tensor(self.getInputs(battle), dtype=torch.float32)

        # Raw network outputs
        logits = self.neuralNetwork(inputs)

        mask, moves = self.translateOutputs(battle)

        if not any(mask):
            log_prob = torch.tensor(0.0)
            self.log_probs.setdefault(battle.battle_tag, []).append(log_prob)
            if self.criticNetwork is not None:
                self.values.setdefault(battle.battle_tag, []).append(
                    self.criticNetwork(inputs)
                )
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
        self.log_probs.setdefault(battle.battle_tag, []).append(log_prob)

        if self.criticNetwork is not None:
            self.values.setdefault(battle.battle_tag, []).append(
                self.criticNetwork(inputs)
            )

        return moves[action.item()]

    @abstractmethod
    def getInputs(self, battle: AbstractBattle) -> list[float]:
        pass

    def translateOutputs(
        self, battle: AbstractBattle
    ) -> tuple[torch.Tensor, list[SingleBattleOrder]]:
        """
        This method will return the moves and switches that can be made

        There should be 14 outputs:
            - 8 for the moves of the current pokemon
                - Alternating between not teraing and teraing
            - 6 for the switches of the team

        Args:
            - battle (AbstractBattle): The current battle

        Returns:
            - tuple[torch.Tensor, list[SingleBattleOrder]]:
                - A tensor of booleans indicating which moves and switches are valid
                - A list of SingleBattleOrder objects corresponding to the moves and switches
        """
        # All the moves plus all the switches
        allOrders = []
        validOrders = []

        # First we add the moves of the active pokemon
        for move in battle.active_pokemon.moves.values():
            allOrders.append(self.create_order(move))
            validOrders.append(move in battle.available_moves)

            allOrders.append(self.create_order(move, terastallize=True))
            validOrders.append(move in battle.available_moves and bool(battle.can_tera))

        # Then we add the switches
        for pokemon in battle.team.values():
            allOrders.append(self.create_order(pokemon))
            validOrders.append(pokemon in battle.available_switches)

        # If there are not enough outputs, we fill with False
        validOrders += [False] * (self.N_OUTPUTS - len(validOrders))
        allOrders += [self.choose_default_move()] * (self.N_OUTPUTS - len(allOrders))

        return torch.tensor(validOrders, dtype=torch.bool), allOrders
