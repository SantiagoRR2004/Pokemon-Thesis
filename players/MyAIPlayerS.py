from poke_env.battle import AbstractBattle
from players import AbstractAIPlayer


class AIPlayerS(AbstractAIPlayer):
    """
    This will have all the information
    """

    N_F_BATTLE = (
        AbstractAIPlayer.N_WEATHERS
        + AbstractAIPlayer.N_FIELDS
        + 1
        + len(AbstractAIPlayer.ENTRY_HAZARDS)
        + len(AbstractAIPlayer.SCREENS_AND_MISC)
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 6
        + len(AbstractAIPlayer.ENTRY_HAZARDS)
        + len(AbstractAIPlayer.SCREENS_AND_MISC)
        + 1
        + 1
        + 6
    )

    def getInputs(self, battle: AbstractBattle) -> list[float]:
        """
        This method will return the inputs for the neural network

        The feature vector will have:
            - The weather conditions (AbstractAIPlayer.N_WEATHERS integers):
                The value is the number of turns the weather has been
                up for. 0 if the weather is not present.
            - The field conditions (AbstractAIPlayer.N_FIELDS integers):
                The value is the number of turns the field has been
                up for. 0 if the field is not present.
            - If the tera can be used (boolean)
            - Entry hazards on our side (len(AbstractAIPlayer.ENTRY_HAZARDS) integers):
                The value is the number of layers of the hazard.
            - Side conditions that are not entry hazards (len(AbstractAIPlayer.SCREENS_AND_MISC) integers):
                The value is the number of turns the condition has been
                up for.
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
            - The opponent's entry hazards (len(AbstractAIPlayer.ENTRY_HAZARDS) integers):
                The value is the number of layers of the hazard.
            - The opponent's side conditions that are not entry hazards (len(AbstractAIPlayer.SCREENS_AND_MISC) integers):
                The value is the number of turns the condition has been
                up for.
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
            - Field conditions that can not happen
                - CRAFTY_SHIELD
                - FIRE_PLEDGE
                - G_MAX_CANNONADE
                - G_MAX_STEELSURGE
                - G_MAX_VINE_LASH
                - G_MAX_VOLCALITH
                - G_MAX_WILDFIRE
                - GRASS_PLEDGE
                - LUCKY_CHANT
                - MATBLOCK
                - QUICK_GUARD
                - WATER_PLEDGE
                - WIDE_GUARD

        Args:
            - battle (AbstractBattle): The current battle

        Returns:
            - list: The inputs for the neural network
        """
        inputs = []

        # Weather conditions
        weather = [0] * self.N_WEATHERS

        for w, nTurns in battle.weather.items():
            weather[w.value - 2] = nTurns - battle.turn

        inputs.extend(weather)

        # Field conditions
        fields = [0] * self.N_FIELDS

        for f, nTurns in battle.fields.items():
            fields[f.value - 2] = nTurns - battle.turn

        inputs.extend(fields)

        # If the tera can be used
        inputs.append(int(battle.can_tera))

        # Entry hazards on our side
        entryHazards = [0] * 4
        for s, nStack in battle.side_conditions.items():
            if self.ENTRY_HAZARDS.get(s.name):
                entryHazards[self.ENTRY_HAZARDS[s.name]] = nStack

        inputs.extend(entryHazards)

        # Side conditions that are not entry hazards
        screens = [0] * len(self.SCREENS_AND_MISC)
        for s, turnStart in battle.side_conditions.items():
            if s.name in self.SCREENS_AND_MISC:
                screens[self.SCREENS_AND_MISC.index(s.name)] = turnStart - battle.turn

        inputs.extend(screens)

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

        # The opponent's entry hazards
        opponentEntryHazards = [0] * 4
        for s, nStack in battle.opponent_side_conditions.items():
            if self.ENTRY_HAZARDS.get(s.name):
                opponentEntryHazards[self.ENTRY_HAZARDS[s.name]] = nStack

        inputs.extend(opponentEntryHazards)

        # The opponent's not entry hazards
        opponentScreens = [0] * len(self.SCREENS_AND_MISC)
        for s, turnStart in battle.opponent_side_conditions.items():
            if s.name in self.SCREENS_AND_MISC:
                opponentScreens[self.SCREENS_AND_MISC.index(s.name)] = (
                    turnStart - battle.turn
                )

        inputs.extend(opponentScreens)

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
