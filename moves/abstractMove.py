from poke_env.battle.effect import _VOLATILE_STATUS_EFFECTS
from abc import ABC, abstractmethod
from poke_env.battle import Move
import pokemonFeatureEncoder


class AbstractMove(ABC):
    """
    This is the abstract class that will be used to represent a move
    as a feature vector for the neural network.
    """

    encoder = pokemonFeatureEncoder.PokemonFeatureEncoder()

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
        "pledgecombo",  # The move is a Pledge Move
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
        "beforeTurnCallback",  # Something happens before the turn starts
        "noparentalbond",  # Doesn't work with Parental Bond
        "futuremove",  # The move is a Future Move (only used by Future Sight)
    }

    @staticmethod
    @abstractmethod
    def getFeatures(move: Move) -> list[float]:
        """
        This method will return the features of the move as a list of floats.

        Args:
            - move (Move): The move to encode.

        Returns:
            - list[float]: The features of the move.
        """
        pass

    N_F_MOVE = 0

    @classmethod
    def getNumberOfFeatures(cls) -> int:
        """
        This method will return the number of features of the move
        """
        if cls.N_F_MOVE == 0:
            raise ValueError("The number of features has not been set yet")
        return cls.N_F_MOVE
