from poke_env.battle import AbstractBattle
from players import AbstractAIPlayer


class AIPlayerS(AbstractAIPlayer):
    """
    This will have all the information
    """

    N_F_BATTLE = 2 + 12

    def getInputs(self, battle: AbstractBattle) -> list[float]:
        """
        This method will return the inputs for the neural network

        The feature vector will have:
            - The index of the active pokemon in the team
            - The player's team (1 + {self.N_F_POKEMON} features per pokemon):
                - Presence indicator (1 if the pokemon is present, 0 otherwise)
                - The encoding of the pokemon (see encodePokemon)
            - The index of the opponent's active pokemon in the opponent's team
            - The opponent's team (1 + {self.N_F_POKEMON} features per pokemon):
                - Presence indicator (1 if the pokemon is present, 0 otherwise)
                - The encoding of the pokemon (see encodePokemon)


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

        Args:
            - battle (AbstractBattle): The current battle

        Missing:
            - battle.can_tera
            - battle.current_observation
            - battle.fields
            - battle.force_switch
            - battle.grounded
            - battle.maybe_trapped
            - battle.observations
            - battle.opponent_side_conditions
            - battle.opponent_used_tera
            - battle.reviving
            - battle.side_conditions
            - battle.trapped
            - battle.turn
            - battle.used_tera
            - battle.weather

        Returns:
            - list: The inputs for the neural network
        """
        inputs = []

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
