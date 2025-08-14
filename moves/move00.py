from moves import AbstractMove


class Move00(AbstractMove):
    """
    This class will be used to represent a move as a feature vector for the neural network.
    """

    N_F_MOVE = 1  # The move id will be the only feature

    @staticmethod
    def getFeatures(move) -> list[float]:
        """
        This method will encode a move into a feature vector

        The feature vector will have:
            - The name of the move (encoded as an integer)

        Args:
            - move (Move): The move to be encoded

        Returns:
            - list[float]: The feature vector of the move
        """
        return [Move00.encoder.encodeMove(move.id)]
