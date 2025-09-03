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

        Args:
            battle (AbstractBattle): The current battle

        Missing:
            - battle.battle_tag
            - battle.can_dynamax
            - battle.can_mega_evolve
            - battle.can_tera
            - battle.can_z_move
            - battle.current_observation
            - battle.dynamax_turns_left
            - battle.fields
            - battle.finished
            - battle.force_switch
            - battle.format
            - battle.gen
            - battle.grounded
            - battle.in_team_preview
            - battle.last_request
            - battle.logger
            - battle.lost
            - battle.max_team_size
            - battle.maybe_trapped
            - battle.observations
            - battle.opponent_dynamax_turns_left
            - battle.opponent_rating
            - battle.opponent_role
            - battle.opponent_side_conditions
            - battle.opponent_used_dynamax
            - battle.opponent_used_mega_evolve
            - battle.opponent_used_tera
            - battle.opponent_used_z_move
            - battle.opponent_username
            - battle.player_role
            - battle.player_username
            - battle.rating
            - battle.reviving
            - battle.rules
            - battle.side_conditions
            - battle.team_size
            - battle.teampreview
            - battle.teampreview_opponent_team
            - battle.teampreview_team
            - battle.trapped
            - battle.turn
            - battle.used_dynamax
            - battle.used_mega_evolve
            - battle.used_tera
            - battle.used_z_move
            - battle.weather
            - battle.won

        Returns:
            list: The inputs for the neural network
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
