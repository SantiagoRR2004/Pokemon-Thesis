from poke_env.ps_client.server_configuration import ServerConfiguration
from pokemons import AbstractPokemon, Pokemon00
from players import AbstractAIPlayer, AIPlayer00
import randomTeams.randomTeam as randomTeam
from moves import AbstractMove, Move00
from critics import AbstractCritic, CriticNetwork01
from actors import AbstractActor, ActorNetwork01
from dotenv import load_dotenv
import otherPlayers
import torch
import torch.nn as nn
import torch.optim as optim
import serverControl
import metricsLogger
import asyncio
import time
import os

load_dotenv()


class Trainer:

    def __init__(
        self,
        *,
        actorClass: AbstractActor,
        playerClass: AbstractAIPlayer,
        moveClass: AbstractMove,
        pokemonClass: AbstractPokemon,
        nTeams: int = float("inf"),
        criticClass: AbstractCritic = None,
        nEpisodes: int = 64,
        fileName: str = None,
        gamma: float = 0.99,
        useRandom: bool = True,
        useMaxDamage: bool = True,
    ) -> None:
        """
        Initialize the Trainer class.

        Args:
            - actor (AbstractActor): The class of actor network to be trained.
            - playerClass (AbstractAIPlayer): The class player for which the actor is trained.
            - moveClass (AbstractMove): The class of move to be used by the player.
            - pokemonClass (AbstractPokemon): The class of pokemon to be used by the player.
            - nTeams (int): Number of teams to use for training. If float("inf"), random teams will be used.
            - criticClass (AbstractCritic, optional): The class of critic network to be trained. If None, only the actor will be trained.
            - nEpisodes (int): Number of episodes to run for training.
            - fileName (str, optional): The name of the file to save the metrics. If None, a default name will be used.
            - gamma (float): Discount factor for future rewards.
            - useRandom (bool): Whether to include a random player as an opponent.
            - useMaxDamage (bool): Whether to include a max damage player as an opponent.

        Returns:
            - None
        """
        self.p = serverControl.startServer()

        self.nEpisodes = int(nEpisodes)
        self.gamma = float(gamma)
        self.useRandom = bool(useRandom)
        self.useMaxDamage = bool(useMaxDamage)
        self.playerClass = playerClass
        self.criticClass = criticClass
        self.fileName = fileName

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
        serverConfig = ServerConfiguration(
            f"ws://localhost:{int(os.getenv("SERVER_PORT"))}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )

        self.args = {
            "max_concurrent_battles": nEpisodes,
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
            self.opponents.append(otherPlayers.getRandomPlayer(self.args))
        if self.useMaxDamage:
            self.opponents.append(otherPlayers.getRandomMaxDamagePlayer(self.args))

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

            # Run n_battles (Episodes)
            for enemy in self.opponents:
                await self.player.battle_against(enemy, n_battles=self.nEpisodes)
            actorLoss = 0
            averageRewardsEpoch = 0
            if self.criticClass:
                criticLoss = 0
                averageCriticRewardsEpoch = 0

            for battle in self.player.battles.values():
                nSteps = battle.turn
                actorLossBattle = 0
                if self.criticClass:
                    criticLossBattle = 0
                finalReward = 1000 if battle.won else -1000

                # Reward sequence
                rewards = [1] * (nSteps - 1) + [finalReward]

                # Compute discounted returns
                discountedRewards = []
                cumulative = 0
                for r in reversed(rewards):
                    cumulative = r + self.gamma * cumulative
                    discountedRewards.insert(0, cumulative)

                averageRewardsEpoch += (
                    torch.tensor(discountedRewards, dtype=torch.float32).mean().item()
                )
                if self.criticClass:
                    averageCriticRewardsEpoch += (
                        torch.stack(self.player.values[battle.battle_tag]).mean().item()
                    )

                # Calculate the loss using actor-critic
                if self.criticClass:
                    for log_prob, G, V in zip(
                        self.player.log_probs[battle.battle_tag],
                        discountedRewards,
                        self.player.values[battle.battle_tag],
                    ):
                        advantage = G - V.item()
                        actorLossBattle += -log_prob * advantage

                    for G, V in zip(
                        discountedRewards, self.player.values[battle.battle_tag]
                    ):
                        criticLossBattle += (V - G).pow(2)
                # Calculate the loss using only actor
                else:
                    for log_prob, G in zip(
                        self.player.log_probs[battle.battle_tag], discountedRewards
                    ):
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
            print(f"{time.time() - start - timeSpent:.2f}", end=" ", flush=True)
            timeSpent = time.time() - start
            secs = (time.time() - start) * (self.nEpochs - (epoch + 1)) / (epoch + 1)
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

            # Reset the server every 50 epochs to avoid memory leaks
            if (epoch + 1) % 50 == 0 and (epoch + 1) != self.nEpochs:
                self.resetServer()

            if (epoch + 1) % 1000 == 0:
                torch.save(self.actor.state_dict(), f"actor_epoch{epoch+1}.pth")
                if self.criticClass:
                    torch.save(self.critic.state_dict(), f"critic_epoch{epoch+1}.pth")

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
        nTeams=float("inf"),
        playerClass=AIPlayer00,
        moveClass=Move00,
        pokemonClass=Pokemon00,
    )

    asyncio.run(t.main())
