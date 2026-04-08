from rewards import AbstractRewardFunction, RewardFunction01
from critics import AbstractCritic, CriticNetwork01
from actors import AbstractActor, ActorNetwork01
from players import AbstractAIPlayer, AIPlayer00
from pokemons import AbstractPokemon, Pokemon00
import randomTeams.randomTeam as randomTeam
from moves import AbstractMove, Move00
import otherPlayers
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import serverControl
import metricsLogger
import asyncio
import random
import time
import gc
import os


def str_to_bool(s: str) -> bool:
    """
    Convert a string to a boolean.

    Args:
        - s (str): The string to convert.

    Returns:
        - bool: The converted boolean.
    """
    if type(s) is bool:
        return s

    s = s.strip().lower()

    if s in {"y", "yes", "t", "true", "on", "1"}:
        return True
    if s in {"n", "no", "f", "false", "off", "0"}:
        return False

    raise ValueError(f"Invalid boolean string: {s}")


class Trainer:

    CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "data", "experiments")
    SELF_PLAY_WINDOW = 20

    def __init__(
        self,
        *,
        actorClass: AbstractActor,
        playerClass: AbstractAIPlayer,
        moveClass: AbstractMove,
        pokemonClass: AbstractPokemon,
        rewardsClass: AbstractRewardFunction,
        nTeams: int = float("inf"),
        criticClass: AbstractCritic = None,
        nEpisodes: int = 64,
        fileName: str = None,
        gamma: float = 0.99,
        useRandom: bool = True,
        useMaxDamage: bool = True,
        useSelfPlay: bool = True,
    ) -> None:
        """
        Initialize the Trainer class.

        Args:
            - actor (AbstractActor): The class of actor network to be trained.
            - playerClass (AbstractAIPlayer): The class player for which the actor is trained.
            - moveClass (AbstractMove): The class of move to be used by the player.
            - pokemonClass (AbstractPokemon): The class of pokemon to be used by the player.
            - rewardsClass (AbstractRewardFunction): The class of reward function to be used for training.
            - nTeams (int): Number of teams to use for training. If float("inf"), random teams will be used.
            - criticClass (AbstractCritic, optional): The class of critic network to be trained. If None, only the actor will be trained.
            - nEpisodes (int): Number of episodes to run for training.
            - fileName (str, optional): The name of the file to save the metrics. If None, a default name will be used.
            - gamma (float): Discount factor for future rewards.
            - useRandom (bool): Whether to include a random player as an opponent.
            - useMaxDamage (bool): Whether to include a max damage player as an opponent.
            - useSelfPlay (bool): Whether to include self-play opponents from recent snapshots.

        Returns:
            - None
        """
        self.p = serverControl.startServer()

        self.nEpisodes = int(nEpisodes)
        self.gamma = float(gamma)
        self.useRandom = str_to_bool(useRandom)
        self.useMaxDamage = str_to_bool(useMaxDamage)
        self.useSelfPlay = str_to_bool(useSelfPlay)
        self.playerClass = playerClass
        self.criticClass = criticClass
        self.rewardsClass = rewardsClass
        self.fileName = fileName
        self.actorClass = actorClass
        self.moveClass = moveClass
        self.pokemonClass = pokemonClass
        self.selfPlaySnapshots: list[tuple[int, dict[str, torch.Tensor]]] = []

        self.setArgs(nTeams=float(nTeams), nEpisodes=self.nEpisodes)
        self.setPlayerArgs(pokemonClass, moveClass, actorClass)

        # Create the opponents
        self.resetOpponents()

        # Create the main player
        self.resetPlayer()

        self.optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        if self.criticClass:
            self.criticOptimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.nEpochs = 1000

        self.createHistoryLists()

        if self.useSelfPlay:
            # Seed with initial policy so self-play is available from epoch 1.
            self.storeSelfPlaySnapshot(epoch=0)

    def createHistoryLists(self) -> None:
        """
        Create the lists to store the training metrics.

        Args:
            - None

        Returns:
            - None
        """
        self.victoryPercentage = []
        self.actorLosses = []
        self.averageRewards = []
        if self.criticClass:
            self.criticLosses = []
            self.averageCriticRewards = []
        self.nTurns = []

    def updateHistoryLists(
        self,
        percentage: float,
        actorLoss: torch.Tensor,
        averageRewardsEpoch: float,
        criticLoss: torch.Tensor = None,
        averageCriticRewardsEpoch: float = None,
    ) -> None:
        """
        Update the lists to store the training metrics.

        Args:
            - percentage (float): Victory percentage for the epoch.
            - actorLoss (torch.Tensor): Actor loss for the epoch.
            - averageRewardsEpoch (float): Average rewards for the epoch.
            - criticLoss (torch.Tensor): Critic loss for the epoch.
            - averageCriticRewardsEpoch (float): Average critic rewards for the epoch.

        Returns:
            - None
        """
        self.victoryPercentage.append(percentage)
        self.actorLosses.append(actorLoss.item())
        self.averageRewards.append(averageRewardsEpoch / len(self.player.battles))

        if self.criticClass:
            self.criticLosses.append(criticLoss.item())
            self.averageCriticRewards.append(
                averageCriticRewardsEpoch / len(self.player.battles)
            )

        self.nTurns.append(
            sum(battle.turn for battle in self.player.battles.values())
            / len(self.player.battles)
        )

    def setArgs(self, nTeams: int, nEpisodes: int) -> None:
        """
        Set the arguments for all the players. The main player
        will use more specific arguments.

        Args:
            - nTeams (int): Number of teams to use for training.
                If float("inf"), random teams will be used.
            - nEpisodes (int): Number of episodes to run for each epoch.

        Returns:
            - None
        """
        serverConfig = serverControl.getServerConfiguration()

        self.args = {
            "max_concurrent_battles": os.cpu_count(),
            "server_configuration": serverConfig,
        }

        # Set the battle format and team
        if nTeams == float("inf"):
            self.args["battle_format"] = "gen9randombattle"
        else:
            self.args["battle_format"] = "gen9purehackmons"
            self.args["team"] = randomTeam.selectRandomTeam(nTeams)

    def setPlayerArgs(
        self,
        pokemonClass: AbstractPokemon,
        moveClass: AbstractMove,
        actorClass: nn.Module,
    ) -> None:
        """
        Set the arguments for the main player.

        The arguments include the actor and critic networks
        and to create those, a dummy player is needed.

        So this function also creates the actor and critic
        instances.

        Args:
            - None

        Returns:
            - None
        """
        playerArgs = self.args.copy()
        playerArgs["pokemonFeatureExtractor"] = pokemonClass(moveClass)

        # Create a dummy player to create the networks
        player: AbstractAIPlayer = self.playerClass(
            network="BlaBlaBla",
            **playerArgs,
        )

        # Create the actor
        self.actor: AbstractActor = actorClass(player)
        playerArgs["network"] = self.actor

        # Create the critic if needed
        if self.criticClass:
            self.critic: AbstractCritic = self.criticClass(player)
            playerArgs["critic"] = self.critic

        # Set the player arguments
        self.playerArgs = playerArgs

        # Delete the dummy player
        del player

    def resetOpponents(self) -> None:
        """
        Reset the opponents list. It uses the flags
        useRandom and useMaxDamage to determine which
        opponents to include.

        Args:
            - None

        Returns:
            - None
        """
        if not hasattr(self, "opponents"):
            self.opponents = []

        for enemy in self.opponents:
            del enemy

        self.opponents = []

        if self.useRandom:
            self.opponents.append(otherPlayers.getRandomPlayer(args=self.args))
        if self.useMaxDamage:
            self.opponents.append(otherPlayers.getRandomMaxDamagePlayer(args=self.args))

    def resetPlayer(self) -> None:
        """
        Reset the main player.

        Args:
            - None

        Returns:
            - None
        """
        if hasattr(self, "player"):
            del self.player

        # We create the player
        self.player: AbstractAIPlayer = self.playerClass(**self.playerArgs)

    def resetServer(self) -> None:
        """
        Reset the server by stopping and starting it again.

        Args:
            - None

        Returns:
            - None
        """
        serverControl.endProcess(self.p)
        self.p.wait()
        self.p = serverControl.startServer()

        # Reset all the players
        self.resetPlayer()
        self.resetOpponents()

        gc.collect()
        torch.cuda.empty_cache()

    def saveMetrics(self) -> None:
        """
        Save the training metrics to a file.

        Args:
            - None

        Returns:
            - None
        """
        kwargs = {
            "victoryPercentage": self.victoryPercentage,
            "actorLosses": self.actorLosses,
            "averageRewards": self.averageRewards,
            "nTurns": self.nTurns,
        }

        # Include critic metrics if applicable
        if self.criticClass:
            kwargs["criticLosses"] = self.criticLosses
            kwargs["averageCriticRewards"] = self.averageCriticRewards

        if self.fileName is not None:
            kwargs["fileName"] = self.fileName

        metricsLogger.saveData(**kwargs)

    def calculateRewards(self, battle) -> list[float]:
        """
        Calculate the rewards for each step in a battle.

        Args:
            - battle: The battle object.

        Returns:
            - List of rewards for each step.
        """
        return self.rewardsClass.calculateRewards(battle)

    async def playBattles(self) -> None:
        """
        Play the battles for the current epoch.

        This is a method to be able to throw an error if all
        battles take too long.

        Args:
            - None

        Returns:
            - None
        """
        for enemy in self.opponents:
            await self.player.battle_against(enemy, n_battles=self.nEpisodes)

        if not self.useSelfPlay:
            return

        sampledEpochs = [
            random.choice(self.selfPlaySnapshots)[0] for _ in range(self.nEpisodes)
        ]

        for snapshotNumber, nBattles in Counter(sampledEpochs).items():

            opponent = self.buildSelfPlayOpponent(
                snapshot_state_dict=self.getSnapshotState(snapshotNumber),
            )

            try:
                await self.player.battle_against(opponent, n_battles=nBattles)
            finally:
                del opponent

    def getSnapshotState(self, snapshotNumber: int) -> dict[str, torch.Tensor]:
        """
        Get a snapshot state dict for a stored epoch.

        Args:
            - snapshotNumber (int): Epoch identifier for the stored snapshot.

        Returns:
            - dict[str, torch.Tensor]: Actor state dict if found.
        """
        for epoch, snapshotState in self.selfPlaySnapshots:
            if epoch == snapshotNumber:
                return snapshotState

        raise ValueError(f"Snapshot for epoch {snapshotNumber} not found.")

    def buildSelfPlayOpponent(
        self,
        *,
        snapshot_state_dict: dict[str, torch.Tensor],
    ) -> AbstractAIPlayer:
        """
        Build an opponent using a frozen historical actor snapshot.

        Args:
            - snapshot_state_dict (dict[str, torch.Tensor]): Saved actor state dict.

        Returns:
            - AbstractAIPlayer: Opponent player instance.
        """
        opponentArgs = self.args.copy()
        opponentArgs["pokemonFeatureExtractor"] = self.pokemonClass(self.moveClass)

        dummyPlayer: AbstractAIPlayer = self.playerClass(
            network="BlaBlaBla",
            **opponentArgs,
        )

        snapshotActor: AbstractActor = self.actorClass(dummyPlayer)
        snapshotActor.load_state_dict(snapshot_state_dict)
        snapshotActor.eval()
        for parameter in snapshotActor.parameters():
            parameter.requires_grad = False

        opponentArgs["network"] = snapshotActor
        opponent: AbstractAIPlayer = self.playerClass(**opponentArgs)

        del dummyPlayer

        return opponent

    def storeSelfPlaySnapshot(self, *, epoch: int) -> None:
        """
        Store a CPU snapshot of the current actor parameters.

        Args:
            - epoch (int): Epoch identifier associated with this snapshot.

        Returns:
            - None
        """
        snapshotState = {
            key: value.detach().cpu().clone()
            for key, value in self.actor.state_dict().items()
        }
        self.selfPlaySnapshots.append((epoch, snapshotState))

        if len(self.selfPlaySnapshots) > self.SELF_PLAY_WINDOW:
            self.selfPlaySnapshots = self.selfPlaySnapshots[-self.SELF_PLAY_WINDOW :]

    def saveSelfPlaySnapshots(self) -> None:
        """
        Save all stored self-play snapshots.

        Args:
            - None

        Returns:
            - None
        """
        if not self.useSelfPlay or not self.selfPlaySnapshots:
            return

        # The current epoch actor is already saved as the main model
        snapshotsToSave = self.selfPlaySnapshots[:-1]
        if not snapshotsToSave:
            return

        snapshotsDirectory = os.path.join(
            self.DATA_DIRECTORY, f"{self.fileName}SelfPlaySnapshots"
        )
        os.makedirs(snapshotsDirectory, exist_ok=True)

        for epoch, snapshotState in snapshotsToSave:
            torch.save(
                snapshotState,
                os.path.join(
                    snapshotsDirectory, f"{self.fileName}ActorEpoch{epoch}.pth"
                ),
            )

    async def main(self) -> None:
        """
        Train a model.

        Args:
            - None

        Returns:
            - None
        """
        start = time.time()
        timeSpent = 0

        for epoch in range(self.nEpochs):

            # Reset the player for new Episodes
            self.player.reset()

            # Reset the battle counters
            self.player.reset_battles()

            startBattles = time.time()

            # Run n_battles (Episodes)
            if epoch == 0:
                # No timeout calculated
                await self.playBattles()

            else:
                # Timeout of 3 times the average time (min 10)
                timeout = max(3 * timeSpent / epoch, 10)

                nTries = 0
                # 3 tries
                while nTries < 3:
                    try:
                        await asyncio.wait_for(self.playBattles(), timeout=timeout)
                        # If successful, break the loop
                        nTries = 3
                    except asyncio.TimeoutError:
                        nTries += 1
                        print(
                            f"Timeout occurred for epoch {epoch+1}. Retrying... ({nTries}/3)",
                            flush=True,
                        )

                        if nTries == 2:
                            # Reset the server on the second timeout
                            self.resetServer()
                        else:
                            # Reset the players
                            self.resetPlayer()
                            self.resetOpponents()

                        # Always set gradients to zero
                        self.optimizer.zero_grad()
                        if self.criticClass:
                            self.criticOptimizer.zero_grad()

                        # Raise the error on the last try
                        if nTries == 3:
                            raise RuntimeError(
                                f"Epoch {epoch+1} failed after 3 tries due to timeouts."
                            )

            endBattles = time.time()

            device = next(self.actor.parameters()).device

            actorLoss = torch.tensor(0.0, device=device)
            averageRewardsEpoch = 0
            if self.criticClass:
                criticLoss = torch.tensor(0.0, device=device)
                averageCriticRewardsEpoch = 0

            for battle in self.player.battles.values():
                nSteps = battle.turn
                actorLossBattle = 0
                if self.criticClass:
                    criticLossBattle = 0

                # Reward sequence
                rewards = self.calculateRewards(battle)

                # Compute discounted returns
                discountedRewards = []
                cumulative = 0
                for r in reversed(rewards):
                    cumulative = r + self.gamma * cumulative
                    discountedRewards.insert(0, cumulative)

                discountedRewardsTensor = torch.tensor(
                    discountedRewards, dtype=torch.float32, device=device
                )

                averageRewardsEpoch += discountedRewardsTensor.mean().item()
                if self.criticClass:
                    averageCriticRewardsEpoch += (
                        torch.stack(self.player.values[battle.battle_tag])
                        .to(device)
                        .mean()
                        .item()
                    )

                # Calculate the loss using actor-critic
                if self.criticClass:
                    for log_prob, G, V in zip(
                        self.player.log_probs[battle.battle_tag],
                        discountedRewardsTensor,
                        self.player.values[battle.battle_tag],
                    ):
                        log_prob = log_prob.squeeze()
                        G = G.squeeze()
                        V = V.squeeze()

                        # Detach critic value so actor and critic backprop use separate graphs.
                        advantage = G - V.detach()
                        actorLossBattle += -log_prob * advantage

                    for G, V in zip(
                        discountedRewardsTensor, self.player.values[battle.battle_tag]
                    ):
                        G = G.squeeze()
                        V = V.squeeze()
                        criticLossBattle += (V - G).pow(2)

                # Calculate the loss using only actor
                else:
                    for log_prob, G in zip(
                        self.player.log_probs[battle.battle_tag],
                        discountedRewardsTensor,
                    ):
                        log_prob = log_prob.squeeze()
                        G = G.squeeze()
                        actorLossBattle += -log_prob * G

                # Normalize the loss by the episodes
                actorLoss += actorLossBattle / nSteps
                if self.criticClass:
                    criticLoss += criticLossBattle / nSteps

            # Normalize the loss by the number of battles
            actorLoss /= len(self.player.battles)
            if self.criticClass:
                criticLoss /= len(self.player.battles)

            # Backpropagation step
            self.optimizer.zero_grad()
            actorLoss.backward()
            self.optimizer.step()

            # Update the critic network
            if self.criticClass:
                self.criticOptimizer.zero_grad()
                criticLoss.backward()
                self.criticOptimizer.step()

            percentage = self.player.n_won_battles / self.player.n_finished_battles

            # We can now print the results of the battles
            print(
                f"{epoch+1:0{len(str(self.nEpochs))}d}/{self.nEpochs}",
                end=" ",
                flush=True,
            )
            trainingEnd = time.time()
            print(
                f"({endBattles - startBattles:.2f}+{trainingEnd - endBattles:.2f})",
                end=" ",
                flush=True,
            )
            print(f"{trainingEnd - start - timeSpent:.2f}", end=" ", flush=True)
            timeSpent = trainingEnd - start
            secs = timeSpent * (self.nEpochs - (epoch + 1)) / (epoch + 1)
            hours, remainder = divmod(int(secs), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"{hours:02}:{minutes:02}:{seconds:02}", end=" ", flush=True)
            print(
                f"{self.fileName} won {percentage*100:.2f}% of battles.",
                end=" ",
                flush=True,
            )
            print()

            # Update the history lists
            self.updateHistoryLists(
                percentage,
                actorLoss,
                averageRewardsEpoch,
                criticLoss if self.criticClass else None,
                averageCriticRewardsEpoch if self.criticClass else None,
            )

            if self.useSelfPlay:
                self.storeSelfPlaySnapshot(epoch=epoch + 1)

            # Reset the server every 25 epochs to avoid memory leaks
            if (epoch + 1) % 25 == 0 and (epoch + 1) != self.nEpochs:
                self.resetServer()

            # Save the model at the last epoch
            if (epoch + 1) == self.nEpochs:
                torch.save(
                    self.actor.state_dict(),
                    os.path.join(self.DATA_DIRECTORY, f"{self.fileName}Actor.pth"),
                )
                if self.criticClass:
                    torch.save(
                        self.critic.state_dict(),
                        os.path.join(self.DATA_DIRECTORY, f"{self.fileName}Critic.pth"),
                    )
                self.saveSelfPlaySnapshots()

        # Save the metrics
        self.saveMetrics()

        # Turn off the server
        serverControl.endProcess(self.p)
        self.p.wait()


if __name__ == "__main__":

    if os.name == "nt":
        # For Windows
        os.system("cls")
    else:
        # For Linux/macOS
        os.system("clear")

    import ignorePokeEnvProblems

    t = Trainer(
        actorClass=ActorNetwork01,
        criticClass=CriticNetwork01,
        rewardsClass=RewardFunction01,
        nTeams=float("inf"),
        playerClass=AIPlayer00,
        moveClass=Move00,
        pokemonClass=Pokemon00,
    )

    asyncio.run(t.main())
