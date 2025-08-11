from poke_env.battle import AbstractBattle
from poke_env.player.battle_order import SingleBattleOrder
from poke_env.battle.effect import _VOLATILE_STATUS_EFFECTS
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
    VOLATILE_STATUS = {}
    for index, status in enumerate(
        sorted(_VOLATILE_STATUS_EFFECTS, key=lambda s: s.name)
    ):
        name = status.name
        VOLATILE_STATUS[name] = index
        VOLATILE_STATUS[name.replace("_", "").lower()] = index

    """
    I think 0 will be untargetable, 1 will be guaranteed, 
    0.5 will be choice and 0.75 will be random.
    """
    MOVE_TARGETS = {
        "ADJACENT_ALLY": [0, 0, 0, 0, 1, 0],
        "ADJACENT_ALLY_OR_SELF": [0, 0, 0, 0.5, 0.5, 0],
        "ADJACENT_FOE": [0.5, 0.5, 0, 0, 0, 0],
        "ALL": [1, 1, 1, 1, 1, 1],
        "ALL_ADJACENT": [1, 1, 0, 0, 1, 0],
        "ALL_ADJACENT_FOES": [1, 1, 0, 0, 0, 0],
        "ALLIES": [0, 0, 0, 1, 1, 1],
        "ALLY_SIDE": [0, 0, 0, 1, 1, 1],
        "ALLY_TEAM": [0, 0, 0, 1, 1, 1],
        "ANY": [0.5, 0.5, 0.5, 0, 0.5, 0.5],
        "FOE_SIDE": [1, 1, 1, 0, 0, 0],
        "NORMAL": [0.5, 0.5, 0, 0, 0.5, 0],
        "RANDOM_NORMAL": [0.75, 0.75, 0, 0, 0, 0],
        "SCRIPTED": [0, 0, 0, 1, 0, 0],
        "SELF": [0, 0, 0, 1, 0, 0],
    }

    OTHER_FLAGS = {
        "protect",  # Blocked by Protect
        "contact",  # Physical Contact
        "mirror",  # Copyable by Mirror Move
        "bullet",  # Bullet-Type
        "punch",  # Punch Move
        "wind",  # Wind Move
        "powder",  # Powder Move
        "snatch",  # Copyable by Snatch
        "reflectable",  # Reflected By Magic Coat/Magic Bounce
        "sound",  # Sound-Type
        "bite",  # Biting Move
        "bypasssub",  # Bypasses Substitute
        "dance",  # Dance Move
        "pulse",  # Aura/pulse move
        "slicing",  # Slicing Move
        "gravity",  # Affected by Gravity
        "basePowerCallback",  # The base power changes based on some condition
        "onBasePower",  # The base power changes based on some condition
        "damageCallback",  # The damage changes based on some condition
        "heal",  # If the move heals the user (It may not have a heal percentage)
        "onAfterHit",  # If the move has an effect after hitting (e.g. knock off)
        "nonsky",  # If the move does not hit in the sky
        "failcopycat",  # If the move fails when copied by Copycat
        "failinstruct",  # The move cannot be repeated by Instruct
        "failmimic",  # The move fails when copied by Mimic
        "nosketch",  # The move cannot be copied by Sketch
        "failmefirst",  # The move fails when used by Me First
        "nosleeptalk",  # The move cannot be used by Sleep Talk
        "charge",  # The move requires a charge turn
        "recharge",  # The move requires a recharge turn
        "cantusetwice",  # The move cannot be used twice in a row
    }
    OTHER_FLAGS_IGNORE = {
        "metronome",  # Not necessary
        "defrost",  # Already in move.thaws_target
        "distance",  # Don't know what it does
        "onModifyMove",  # Don't know what it does
        "mustpressure",  # Don't know what it does
        "allyanim",  # Don't know what it does
        "onPrepareHit",  # Don't know what it does
        "failencore",  # Think this is only used for the move Encore
        "noassist",  # Don't know what it does
        "onTry",  # Don't know what it does
        "onHit",  # Something happens on hit
        "onTryHit",  # Something happens on try hit
        "onAfterMove",  # Something happens after the move is used
        "onTryMove",  # Something happens when the move is tried
        "onMoveFail",  # Something happens when the move fails
        "onEffectiveness",  # Seems to be related to Freeze Dry
        "onHitField",  # Something happens to the field when the move is used
        "onAfterMoveSecondarySelf",  # Onlyused by Relic Song
        "beforeMoveCallback",  # Something happens before the move is used
        "noparentalbond",  # Doesn't work with Parental Bond
        "futuremove",  # The move is a Future Move (only used by Future Sight)
    }

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
