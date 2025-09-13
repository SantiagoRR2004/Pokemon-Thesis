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


async def main(
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
) -> None:
    """
    Train an actor-critic model for a given number of episodes.

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

    Returns:
        - None
    """
    p = serverControl.startServer()

    serverConfig = ServerConfiguration(
        f"ws://localhost:{int(os.getenv("SERVER_PORT"))}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    nTeams = float(nTeams)
    nEpisodes = int(nEpisodes)
    gamma = float(gamma)
    useRandom = bool(useRandom)

    args = {
        "max_concurrent_battles": nEpisodes,
        "server_configuration": serverConfig,
    }
    if nTeams == float("inf"):
        args["battle_format"] = "gen9randombattle"
    else:
        args["battle_format"] = "gen9purehackmons"
        args["team"] = randomTeam.selectRandomTeam(nTeams)

    playerArgs = args.copy()
    playerArgs["pokemonFeatureExtractor"] = pokemonClass(moveClass)

    player: AbstractAIPlayer = playerClass(
        network="BlaBlaBla",
        **playerArgs,
    )

    opponents = []
    if useRandom:
        opponents.append(otherPlayers.getRandomPlayer(args))

    # # We create the random player
    # second_player = otherPlayers.getRandomPlayer(args)

    actor = actorClass(player)
    player.neuralNetwork = actor
    playerArgs["network"] = actor

    if criticClass:
        critic = criticClass(player)
        player.criticNetwork = critic
        playerArgs["critic"] = critic
    else:
        critic = None

    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    if criticClass:
        criticOptimizer = optim.Adam(critic.parameters(), lr=1e-3)

    nEpochs = 1000

    victoryPercentage = []
    actorLosses = []
    averageRewards = []
    if criticClass:
        criticLosses = []
        averageCriticRewards = []
    nTurns = []

    start = time.time()

    for epoch in range(nEpochs):

        # Reset the player for new Episodes
        player.reset()

        # Reset the battle counters
        player.reset_battles()

        # Run n_battles (Episodes)
        for enemy in opponents:
            await player.battle_against(enemy, n_battles=nEpisodes)

        actorLoss = 0
        averageRewardsEpoch = 0
        if criticClass:
            criticLoss = 0
            averageCriticRewardsEpoch = 0

        for battle in player.battles.values():
            nSteps = battle.turn
            actorLossBattle = 0
            if criticClass:
                criticLossBattle = 0
            finalReward = 1000 if battle.won else -1000

            # Reward sequence
            rewards = [1] * (nSteps - 1) + [finalReward]

            # Compute discounted returns
            discountedRewards = []
            cumulative = 0
            for r in reversed(rewards):
                cumulative = r + gamma * cumulative
                discountedRewards.insert(0, cumulative)

            averageRewardsEpoch += (
                torch.tensor(discountedRewards, dtype=torch.float32).mean().item()
            )
            if criticClass:
                averageCriticRewardsEpoch += (
                    torch.stack(player.values[battle.battle_tag]).mean().item()
                )

            # Calculate the loss using actor-critic
            if criticClass:
                for log_prob, G, V in zip(
                    player.log_probs[battle.battle_tag],
                    discountedRewards,
                    player.values[battle.battle_tag],
                ):
                    advantage = G - V.item()
                    actorLossBattle += -log_prob * advantage

                for G, V in zip(discountedRewards, player.values[battle.battle_tag]):
                    criticLossBattle += (V - G).pow(2)
            # Calculate the loss using only actor
            else:
                for log_prob, G in zip(
                    player.log_probs[battle.battle_tag], discountedRewards
                ):
                    actorLossBattle += -log_prob * G

            # Normalize the loss by the episodes
            actorLoss += actorLossBattle / nSteps
            if criticClass:
                criticLoss += criticLossBattle / nSteps

        # Normalize the loss by the number of battles
        actorLoss /= len(player.battles)
        if criticClass:
            criticLoss /= len(player.battles)

        # Backpropagation step
        optimizer.zero_grad()
        actorLoss.backward()
        optimizer.step()

        # Update the critic network
        if criticClass:
            criticOptimizer.zero_grad()
            criticLoss.backward()
            criticOptimizer.step()

        percentage = player.n_won_battles / player.n_finished_battles

        # We can now print the results of the battles
        print(f"{epoch+1:0{len(str(nEpochs))}d}/{nEpochs}", end=" ", flush=True)
        secs = (time.time() - start) * (nEpochs - (epoch + 1)) / (epoch + 1)
        hours, remainder = divmod(int(secs), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"{hours:02}:{minutes:02}:{seconds:02}", end=" ", flush=True)
        print(
            f"{fileName} won {percentage*100:.2f}% of battles.",
            end=" ",
            flush=True,
        )
        print()

        victoryPercentage.append(percentage)
        actorLosses.append(actorLoss.item())
        averageRewards.append(averageRewardsEpoch / len(player.battles))
        if criticClass:
            criticLosses.append(criticLoss.item())
            averageCriticRewards.append(averageCriticRewardsEpoch / len(player.battles))
        nTurns.append(
            sum(battle.turn for battle in player.battles.values()) / len(player.battles)
        )

        if (epoch + 1) % 100 == 0 and (epoch + 1) != nEpochs:
            serverControl.endProcess(p)
            p.wait()
            p = serverControl.startServer()

            del player

            for enemy in opponents:
                del enemy

            # We create the player again
            player = playerClass(**playerArgs)

            opponents = []
            if useRandom:
                opponents.append(otherPlayers.getRandomPlayer(args))

            # # We create the random player
            # second_player = otherPlayers.getRandomPlayer(args)

        if (epoch + 1) % 1000 == 0:
            torch.save(actor.state_dict(), f"actor_epoch{epoch+1}.pth")
            if criticClass:
                torch.save(critic.state_dict(), f"critic_epoch{epoch+1}.pth")

    # Save the metrics
    kwargs = {
        "victoryPercentage": victoryPercentage,
        "actorLosses": actorLosses,
        "averageRewards": averageRewards,
        "nTurns": nTurns,
    }

    if criticClass:
        kwargs["criticLosses"] = criticLosses
        kwargs["averageCriticRewards"] = averageCriticRewards

    if fileName is not None:
        kwargs["fileName"] = fileName

    metricsLogger.saveData(**kwargs)

    # Turn off the server
    serverControl.endProcess(p)
    p.wait()


if __name__ == "__main__":

    if os.name == "nt":
        # For Windows
        os.system("cls")
    else:
        # For Linux/macOS
        os.system("clear")

    import ignorePokeEnvProblems

    asyncio.run(
        main(
            actorClass=ActorNetwork01,
            criticClass=CriticNetwork01,
            nTeams=float("inf"),
            playerClass=AIPlayer00,
            moveClass=Move00,
            pokemonClass=Pokemon00,
        )
    )
