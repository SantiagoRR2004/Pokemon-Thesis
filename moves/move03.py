from poke_env.battle.effect import _VOLATILE_STATUS_EFFECTS
from moves import AbstractMove


class Move03(AbstractMove):
    """
    This class will be used to represent a move as a feature vector for the neural network.
    """

    N_F_MOVE = (
        1
        + AbstractMove.encoder.N_F_TYPES
        + 3
        + 1
        + 1
        + 1
        + 1
        + 1
        + 6
        + 1
        + 3
        + len(AbstractMove.encoder.BOOSTABLE_STATS)
        + 1
        + 1
    )

    @staticmethod
    def getFeatures(move) -> list[float]:
        """
        This method will encode a move into a feature vector

        The feature vector will have:
            - The name of the move (encoded as an integer)
            - The type of the move ({self.encoder.N_F_TYPES} One-Hot Encoding)
            - The category of the move (3 One-Hot Encoding)
                - PHYSICAL
                - SPECIAL
                - STATUS
            - The base power of the move (normalized to a float between 0 and 1)
                Divided by 120 because higher than that and they become very rare.
            - The accuracy of the move (a float between 0 and 1)
                It is already given between 0 and 1.
            - The remaining PP percentage of the move
            - The maximum PP of the move (a float between 0 and 1)
                We use the base PP without using PP up
            - The priority of the move (float between -1 to 1)
            - Where the move targets (6 floats between 0 and 1)
            - The critical hit ratio (float between 0 and 1)
            - Number of hits (3 floats between 0 and 1):
                - The minimum number of hits
                - The expected number of hits
                - The maximum number of hits
            - The boosts the move gives ({len(self.BOOSTABLE_STATS)} floats between -1 and 1):
                - Attack
                - Defense
                - Special Attack
                - Special Defense
                - Speed
                - Accuracy
                - Evasion
            - The protection data (1 boolean):
                - If the move increases the protect counter (1 if true, 0 otherwise)
            - If the move creates something on the battlefield (1 boolean):
                - If the move creates a side condition (1 if true, 0 otherwise)

        Args:
            - move (Move): The move to be encoded

        Returns:
            - list[float]: The feature vector of the move
        """
        toret = []
        toret.append(AbstractMove.encoder.encodeMove(move.id))

        types = [0] * AbstractMove.encoder.N_F_TYPES
        types[move.type.value - 1] = 1
        toret += types

        # Add the category of the move
        moveCategory = [0] * 3
        moveCategory[move.category.value - 1] = 1
        toret.extend(moveCategory)

        # Add the base power of the move
        toret.append(move.base_power / 120)

        # Add the accuracy of the move
        toret.append(move.accuracy)

        # Add the PP of the move
        toret.append(move.current_pp / move.max_pp)
        toret.append(move.max_pp * 0.625 / 40)

        # Add the priority of the move
        toret.append(move.priority / 8)

        # Add the target of the move
        toret.extend(AbstractMove.MOVE_TARGETS[move.target.name])

        # Add the critical hit ratio
        toret.append(move.crit_ratio / 6)

        # The number of hits
        toret.append(move.n_hit[0] / 5)
        toret.append(move.expected_hits / 5)
        toret.append(move.n_hit[1] / 5)

        # The boosts the move gives or takes to self
        if move.boosts or move.self_boost:
            boostsCombined = {**(move.boosts or {}), **(move.self_boost or {})}
            for stat in AbstractMove.encoder.BOOSTABLE_STATS:
                toret.append(boostsCombined.get(stat, 0) / 2)
        else:
            # If there are no boosts we add zeros
            toret.extend([0] * len(AbstractMove.encoder.BOOSTABLE_STATS))

        ## The protection data
        # If the move increases the protect counter
        toret.append(int(move.is_protect_counter))

        ## The move creates something on the battlefield
        # If the move creates a side condition
        toret.append(1 if move.side_condition else 0)

        return toret
