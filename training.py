from poke_env.ps_client.server_configuration import ServerConfiguration
from pokemons import AbstractPokemon, Pokemon00
from players import AbstractAIPlayer, AIPlayer00
import randomTeams.randomTeam as randomTeam
from moves import AbstractMove, Move00
from critics import CriticNetwork01
from actors import ActorNetwork01
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
        actorClass: nn.Module,
        playerClass: AbstractAIPlayer,
        moveClass: AbstractMove,
        pokemonClass: AbstractPokemon,
        nTeams: int = float("inf"),
        criticClass: nn.Module = None,
        nEpisodes: int = 64,
        fileName: str = None,
        gamma: float = 0.99,
        useRandom: bool = True,
        useMaxDamage: bool = True,
    ) -> None:
        """
        Initialize the Trainer class.

        Args:
            - actor (nn.Module): The class of actor network to be trained.
            - playerClass (AbstractAIPlayer): The class player for which the actor is trained.
            - moveClass (AbstractMove): The class of move to be used by the player.
            - pokemonClass (AbstractPokemon): The class of pokemon to be used by the player.
            - nTeams (int): Number of teams to use for training. If float("inf"), random teams will be used.
            - criticClass (nn.Module, optional): The class of critic network to be trained. If None, only the actor will be trained.
            - nEpisodes (int): Number of episodes to run for training.
            - fileName (str, optional): The name of the file to save the metrics. If None, a default name will be used.
            - gamma (float): Discount factor for future rewards.
            - useRandom (bool): Whether to include a random player as an opponent.
            - useMaxDamage (bool): Whether to include a max damage player as an opponent.

        Returns:
            - None
        """
        self.p = serverControl.startServer()

        serverConfig = ServerConfiguration(
            f"ws://localhost:{int(os.getenv("SERVER_PORT"))}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )

        nTeams = float(nTeams)
        self.nEpisodes = int(nEpisodes)
        self.gamma = float(gamma)
        self.useRandom = bool(useRandom)
        self.useMaxDamage = bool(useMaxDamage)
        self.playerClass = playerClass
        self.criticClass = criticClass
        self.fileName = fileName

        self.args = {
            "max_concurrent_battles": nEpisodes,
            "server_configuration": serverConfig,
        }
        if nTeams == float("inf"):
            self.args["battle_format"] = "gen9randombattle"
        else:
            self.args["battle_format"] = "gen9purehackmons"
            self.args["team"] = randomTeam.selectRandomTeam(nTeams)

        self.playerArgs = self.args.copy()
        self.playerArgs["pokemonFeatureExtractor"] = pokemonClass(moveClass)

        self.player: AbstractAIPlayer = self.playerClass(
            network="BlaBlaBla",
            **self.playerArgs,
        )

        self.opponents = []
        if self.useRandom:
            self.opponents.append(otherPlayers.getRandomPlayer(self.args))
        if self.useMaxDamage:
            self.opponents.append(otherPlayers.getRandomMaxDamagePlayer(self.args))

        self.actor = actorClass(self.player)
        self.player.neuralNetwork = self.actor
        self.playerArgs["network"] = self.actor

        if criticClass:
            self.critic = criticClass(self.player)
            self.player.criticNetwork = self.critic
            self.playerArgs["critic"] = self.critic
        else:
            self.critic = None

        self.optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        if self.criticClass:
            self.criticOptimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.nEpochs = 1000

        self.victoryPercentage = []
        self.actorLosses = []
        self.averageRewards = []
        if self.criticClass:
            self.criticLosses = []
            self.averageCriticRewards = []
        self.nTurns = []

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

            if (epoch + 1) % 10 == 0 and (epoch + 1) != self.nEpochs:
                serverControl.endProcess(self.p)
                self.p.wait()
                self.p = serverControl.startServer()

                del self.player

                for enemy in self.opponents:
                    del enemy

                # We create the player again
                self.player = self.playerClass(**self.playerArgs)

                self.opponents = []
                if self.useRandom:
                    self.opponents.append(otherPlayers.getRandomPlayer(self.args))
                if self.useMaxDamage:
                    self.opponents.append(
                        otherPlayers.getRandomMaxDamagePlayer(self.args)
                    )

            if (epoch + 1) % 1000 == 0:
                torch.save(self.actor.state_dict(), f"actor_epoch{epoch+1}.pth")
                if self.criticClass:
                    torch.save(self.critic.state_dict(), f"critic_epoch{epoch+1}.pth")

        # Save the metrics
        kwargs = {
            "victoryPercentage": self.victoryPercentage,
            "actorLosses": self.actorLosses,
            "averageRewards": self.averageRewards,
            "nTurns": self.nTurns,
        }

        if self.criticClass:
            kwargs["criticLosses"] = self.criticLosses
            kwargs["averageCriticRewards"] = self.averageCriticRewards

        if self.fileName is not None:
            kwargs["fileName"] = self.fileName

        metricsLogger.saveData(**kwargs)

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
