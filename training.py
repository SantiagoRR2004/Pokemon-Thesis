from MyAIPlayer import AIPlayer
import serverControl
import asyncio
import neat
import os


async def playGame(
    neuralNetwork1: neat.nn.FeedForwardNetwork,
    neuraNetwork2: neat.nn.FeedForwardNetwork,
) -> int:
    """
    This is the function that checks if a neural network is better than another one

    Args:
        neuralNetwork1 (neat.nn.FeedForwardNetwork): The first neural network
        neuraNetwork2 (neat.nn.FeedForwardNetwork): The second neural network

    Returns:
        int: 1 if the first neural network is better, 0 otherwise
    """
    player1 = AIPlayer(battle_format="gen9randombattle", network=neuralNetwork1)
    player2 = AIPlayer(network=neuraNetwork2)

    await player1.battle_against(player2, n_battles=1)

    if player1.n_won_battles > player2.n_won_battles:
        return 1
    else:
        return 0


async def evaluate_genomes(genomes, config):
    """
    Evaluates all genomes in the population by playing them against each other.
    """
    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness score

    # Play games between random pairs of genomes
    for i in range(len(genomes)):
        for j in range(i + 1, len(genomes)):  # Ensure each pair plays only once
            genome_id1, genome1 = genomes[i]
            genome_id2, genome2 = genomes[j]

            # Create NEAT neural networks from genomes
            net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
            net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

            # Play a game
            result = await playGame(net1, net2)

            # Assign fitness points
            if result == 1:
                genome1.fitness += 1  # Winning network gets a point
            else:
                genome2.fitness += 1  # Losing network gets a point


def evaluate_genomes_wrapper(genomes, config):
    # We need to run the async function inside an event loop
    return asyncio.run(evaluate_genomes(genomes, config))


def run_neat(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    population = neat.Population(config)

    # Add statistics reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT
    winner = population.run(evaluate_genomes_wrapper, 1)  # Run for x generations

    print("\nBest genome:\n{}".format(winner))


def main():
    serverControl.startServer()

    run_neat("config-feedforward")


if __name__ == "__main__":

    if os.name == "nt":
        # For Windows
        os.system("cls")
    else:
        # For Linux/macOS
        os.system("clear")

    import ignorePokeEnvProblems

    main()
