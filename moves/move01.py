from moves import AbstractMove


class Move01(AbstractMove):
    """
    This class will be used to represent a move as a feature vector for the neural network.
    """

    N_F_MOVE = 1 + AbstractMove.encoder.N_F_TYPES + 3 + 1 + 1

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

        return toret
