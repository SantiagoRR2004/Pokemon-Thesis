from poke_env.battle import AbstractBattle
from players import AbstractAIPlayer


class AIPlayerS(AbstractAIPlayer):
    """
    This will have all the information
    """

    N_F_BATTLE = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 + 1 + 1 + 6

    def getInputs(self, battle: AbstractBattle) -> list[float]:
        """
        This method will return the inputs for the neural network

        The feature vector will have:
            - If the tera can be used (boolean)
            - If the user has to select a pokemon to switch (boolean)
            - If the user has to select a pokemon to revive (boolean)
            - If the pokemon might be trapped (boolean)
            - If the pokemon is trapped (boolean)
            - If the user's tera has been used (boolean)
            - The user's active pokemon is grounded (boolean)
            - The index of the active pokemon in the team (integer)
            - The player's team (1 + {self.N_F_POKEMON} features per pokemon):
                - Presence indicator (1 if the pokemon is present, 0 otherwise)
                - The encoding of the pokemon
            - If the opponent's tera has been used (boolean)
            - The index of the opponent's active pokemon in the opponent's team (integer)
            - The opponent's team (1 + {self.N_F_POKEMON} features per pokemon):
                - Presence indicator (1 if the pokemon is present, 0 otherwise)
                - The encoding of the pokemon


        The following are not used:
            - Anything to do with mega evolution
                - battle.can_mega_evolve
                - battle.opponent_used_mega_evolve
                - battle.used_mega_evolve
            - Anything to do with z-moves
                - battle.can_z_move
                - battle.opponent_used_z_move
                - battle.used_z_move
            - Anything to do with dynamax
                - battle.can_dynamax
                - battle.dynamax_turns_left
                - battle.opponent_dynamax_turns_left
                - battle.opponent_used_dynamax
                - battle.used_dynamax
            - External to the battle
                - battle.battle_tag
                - battle.finished
                - battle.last_request
                - battle.logger
                - battle.lost
                - battle.opponent_role
                - battle.opponent_username
                - battle.player_role
                - battle.player_username
                - battle.rules
                - battle.won
            - Can't train with ELO
                - battle.rating
                - battle.opponent_rating
            - Only training for one format
                - battle.format
                - battle.gen
                - battle.team_size
            - No team preview in randbats
                - battle.in_team_preview
                - battle.max_team_size
                - battle.teampreview
                - battle.teampreview_opponent_team
                - battle.teampreview_team
            - battle.current_observation
                All the information here is already
                in other attributes of the battle object.
            - battle.observations
                It is a dictionary that contains the
                current_observation for all turns of the battle.

        Args:
            - battle (AbstractBattle): The current battle

        Missing:
            - battle.fields
            - battle.opponent_side_conditions
            - battle.side_conditions
            - battle.turn
            - battle.weather

        Returns:
            - list: The inputs for the neural network
        """
        inputs = []

        # If the tera can be used
        inputs.append(int(battle.can_tera))

        # If the user has to select a pokemon to switch
        inputs.append(int(battle.force_switch))

        # If the pokemon is reviving
        inputs.append(int(battle.reviving))

        # If the pokemon might be trapped
        inputs.append(int(battle.maybe_trapped))

        # If the pokemon is trapped
        inputs.append(int(battle.trapped))

        # If the user's tera has been used
        inputs.append(int(battle.used_tera))

        # If the pokemon is grounded
        inputs.append(int(battle.grounded))

        # Which pokemon is active
        try:
            index = list(battle.team.values()).index(battle.active_pokemon) + 1
        except ValueError:
            index = 0
        inputs.append(index)

        # First our team
        for pokemon in battle.team.values():
            inputs += [1] + self.pokemonFeatureExtractor.getFeatures(pokemon, battle)
        # Fill with zeros if unknown pokemon
        inputs += (
            [0]
            * (6 - len(battle.team))
            * (1 + self.pokemonFeatureExtractor.getNumberOfFeatures())
        )

        # If the opponent's tera has been used
        inputs.append(int(battle.opponent_used_tera))

        # Which opponent's pokemon is active
        try:
            index = (
                list(battle.opponent_team.values()).index(
                    battle.opponent_active_pokemon
                )
                + 1
            )
        except ValueError:
            index = 0
        inputs.append(index)

        # The opponent's team
        for pokemon in battle.opponent_team.values():
            inputs += [1] + self.pokemonFeatureExtractor.getFeatures(pokemon, battle)
        # Fill with zeros if unknown pokemon
        inputs += (
            [0]
            * (6 - len(battle.opponent_team))
            * (1 + self.pokemonFeatureExtractor.getNumberOfFeatures())
        )

        return inputs
