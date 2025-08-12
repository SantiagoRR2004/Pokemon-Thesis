from poke_env.battle.effect import _VOLATILE_STATUS_EFFECTS
from moves import AbstractMove


class MoveS(AbstractMove):
    """
    This class will be used to represent a move as a feature vector for the neural network.
    """

    @staticmethod
    def getFeatures(move) -> list[float]:
        """
        This method will encode a move into a feature vector

        The feature vector will have:
            - The name of the move (encoder.NUM_UNIQUE_MOVES One-Hot Encoding)
            - The type of the move ({self.N_F_TYPES} One-Hot Encoding)
            - The category of the move (3 One-Hot Encoding)
                - PHYSICAL
                - SPECIAL
                - STATUS
            - The defense the move targets (3 One-Hot Encoding):
                - PHYSICAL
                - SPECIAL
                - STATUS
            - The base power of the move (normalized to a float between 0 and 1)
                Divided by 120 because higher than that and they become very rare.
            - If the base power is preset or level dependent (boolean)
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
            - The protection data (3 booleans):
                - If the move is a protect move (1 if true, 0 otherwise)
                - If the move increases the protect counter (1 if true, 0 otherwise)
                - If it is a side protect move (1 if true, 0 otherwise)
            - If the move creates something on the battlefield (5 booleans):
                - If the move is a weather move (1 if true, 0 otherwise)
                - If the move is a pseudo weather move (1 if true, 0 otherwise)
                - If the move creates a side condition (1 if true, 0 otherwise)
                - If the move creates a terrain (1 if true, 0 otherwise)
                - If the move heals a switching partner (1 if true, 0 otherwise)
            - If it gives a normal status condition (6 floats between 0 and 1):
                It can be the main purpose of the move or a secondary effect,
                so it is the chance of inflicting the status, once the move hits.
            - If it gives a volitile status condition ({len(self._VOLATILE_STATUS_EFFECTS)} between 0 and 1):
                It can be the main purpose of the move or a secondary effect,
                so it is the chance of inflicting the volitile status
                 once the move hits.
            - Guaranteed secondary effects:
                - If the move can break trough protect (1 if true, 0 otherwise)
                - The percentage the move drains (float between 0 and 1)
                - The percentage the move heals (float between 0 and 1)
                - The percentage of recoil damage (float between 0 and 1)
                - If the move is a self-switch move (1 if true, 0 otherwise)
                - The move forces the opponent to switch (1 if true, 0 otherwise)
                - If the move ignores abilities (1 if true, 0 otherwise)
                - If the move ignores defensive boosts (1 if true, 0 otherwise)
                - If the move ignores evasion (1 if true, 0 otherwise)
                - If the move ignores immunities (1 if true, 0 otherwise)
                - If the move is a self-destruct move (1 if true, 0 otherwise)
                - If the move thaws (1 if true, 0 otherwise)
            - If the move is a stalling move (boolean) (Categorized by Showdown)
            - The boosts the move can give to self in the secondary effects
                ({len(self.BOOSTABLE_STATS)} floats between -1 and 1)
            - The boosts the move can give to opponent in the secondary effects
                ({len(self.BOOSTABLE_STATS)} floats between -1 and 1)
            - Other flags (len(self.OTHER_FLAGS) integers)
                They are simply what it is stored inside and didn't get
                added to self.OTHER_FLAGS_IGNORE.

        The following features are not used:
            - Anything related to z-moves
            - Anything related to dynamax
            - move.is_empty (Seems to always be False)
            - move.non_ghost_target (It is only used for curse)
            - move.no_pp_boosts (It is only used for revival blessing)
            - move.sleep_usable (It is only used for Sleep Talk and Snore)
            - move.steals_boosts (It is only used for Spectral Thief)
            - move.request_target (Seems to be not used)
            - move.use_target_offensive (It is only used for Foul Play)
            - move.entry (Everything here is somewhere else)
            - move.target
            - move.secondary["onHit"] This seems to be an error in the poke_env library
                It says the change of some status or volatile status, but there is no way to know
                which one it is.
            - move.deduced_target (Basically the same as move.target)

        Args:
            - move (Move): The move to be encoded

        Returns:
            - list[float]: The feature vector of the move
        """
        toret = []
        toret.extend(MoveS.encoder.encodeMoveList(move.id))

        types = [0] * MoveS.N_F_TYPES
        types[move.type.value - 1] = 1
        toret += types

        # Add the category of the move
        moveCategory = [0] * 3
        moveCategory[move.category.value - 1] = 1
        toret.extend(moveCategory)

        # Add the defense the move targets
        defensiveCategory = [0] * 3
        defensiveCategory[move.defensive_category.value - 1] = 1
        toret.extend(defensiveCategory)

        # Add the base power of the move
        toret.append(move.base_power / 120)

        # Add if the base power is preset or level dependent
        toret.append(1 if move.damage else 0)

        # Add the accuracy of the move
        toret.append(move.accuracy)

        # Add the PP of the move
        toret.append(move.current_pp / move.max_pp)
        toret.append(move.max_pp * 0.625 / 40)

        # Add the priority of the move
        toret.append(move.priority / 8)

        # Add the target of the move
        toret.extend(MoveS.MOVE_TARGETS[move.target.name])

        # Add the critical hit ratio
        toret.append(move.crit_ratio / 6)

        # The number of hits
        toret.append(move.n_hit[0] / 5)
        toret.append(move.expected_hits / 5)
        toret.append(move.n_hit[1] / 5)

        # The boosts the move gives or takes to self
        if move.boosts or move.self_boost:
            boostsCombined = {**(move.boosts or {}), **(move.self_boost or {})}
            for stat in MoveS.BOOSTABLE_STATS:
                toret.append(boostsCombined.get(stat, 0) / 2)
        else:
            # If there are no boosts we add zeros
            toret.extend([0] * len(MoveS.BOOSTABLE_STATS))

        ## The protection data
        # If the move is a protect move
        toret.append(int(move.is_protect_move))

        # If the move increases the protect counter
        toret.append(int(move.is_protect_counter))

        # If it is a side protect move
        toret.append(int(move.is_side_protect_move))

        ## The move creates something on the battlefield
        # If the move is a weather move
        toret.append(1 if move.weather else 0)

        # If the move is a pseudo weather move
        toret.append(1 if move.pseudo_weather else 0)

        # If the move creates a side condition
        toret.append(1 if move.side_condition else 0)

        # If the move creates a terrain
        toret.append(1 if move.terrain else 0)

        # If the move heals a switching partner
        toret.append(1 if move.slot_condition else 0)

        # The status the move tries to inflict
        statusToret = [0] * len(MoveS.STATUS)

        ## The status the move tries to inflict
        if move.status:
            statusToret[MoveS.STATUS.index(move.status.name.lower())] = 1

        if move.secondary:
            for s in move.secondary:

                if s.get("status"):
                    statusToret[MoveS.STATUS.index(s["status"])] = s["chance"] / 100

        toret.extend(statusToret)

        ## The volitile status the move tries to inflict
        volatileStatusToret = [0] * len(_VOLATILE_STATUS_EFFECTS)

        if move.volatile_status:
            volatileStatusToret[MoveS.VOLATILE_STATUS[move.volatile_status.name]] = 1

        if move.secondary:
            for s in move.secondary:

                if s.get("volatileStatus"):
                    volatileStatusToret[MoveS.VOLATILE_STATUS[s["volatileStatus"]]] = (
                        s["chance"] / 100
                    )

        toret.extend(volatileStatusToret)

        ## Guaranteed secondary effects
        # If the move can break trough protect
        toret.append(int(move.breaks_protect))

        # The percentage the move drains
        toret.append(move.drain)

        # The percentage the move heals
        toret.append(move.heal)

        # The percentage of recoil damage
        toret.append(move.recoil)

        # If the move is a self-switch move
        toret.append(1 if move.self_switch else 0)

        # The move forces the opponent to switch
        toret.append(int(move.force_switch))

        # If the move ignores abilities
        toret.append(int(move.ignore_ability))

        # If the move ignores defensive boosts
        toret.append(int(move.ignore_defensive))

        # If the move ignores evasion
        toret.append(int(move.ignore_evasion))

        # If the move ignores immunities
        toret.append(1 if move.ignore_immunity else 0)

        # If the move is a self-destruct move
        toret.append(1 if move.self_destruct else 0)

        # If the move thaws
        toret.append(int(move.thaws_target))

        # If the move is a stalling move
        toret.append(int(move.stalling_move))

        ## The lists that could be in secondary
        boostSelf = [0] * len(MoveS.BOOSTABLE_STATS)
        boostOpponent = [0] * len(MoveS.BOOSTABLE_STATS)

        # The secondary effects of the move
        if move.secondary:
            for s in move.secondary:

                if s.get("boosts"):
                    for b in s["boosts"]:
                        boostOpponent[MoveS.BOOSTABLE_STATS.index(b)] = (
                            s["boosts"][b] / 2
                        )

                elif s.get("self"):
                    if s["self"].get("boosts"):
                        for b in s["self"]["boosts"]:
                            boostSelf[MoveS.BOOSTABLE_STATS.index(b)] = (
                                s["self"]["boosts"][b] / 2
                            )

                elif s.get("onHit") or s.get("status") or s.get("volatileStatus"):
                    pass

                else:
                    pass

        toret.extend(boostSelf)
        toret.extend(boostOpponent)

        ## Other flags
        for f in MoveS.OTHER_FLAGS:
            toret.append(int(f in move.flags))

        return toret
