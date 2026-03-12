from poke_env.ps_client.server_configuration import ServerConfiguration
from dotenv import load_dotenv
import subprocess
import signal
import atexit
import time
import os

load_dotenv()


def endProcess(process: subprocess.Popen) -> None:
    """
    Ends the given process.

    Args:
        - process (subprocess.Popen): The process to end

    Returns:
        - None
    """
    if process.poll() is not None:
        return

    # Step 1: Send a SIGINT (interrupt) signal to the entire process group
    os.killpg(os.getpgid(process.pid), signal.SIGINT)

    # Step 2: Wait for the process to finish
    process.wait()

    # Reset the created players set
    from otherPlayers import createdPlayers

    createdPlayers.clear()

    print("Server shut down successfully.", flush=True)


def startServer() -> subprocess.Popen:
    """
    Starts the Pokemon Showdown server.

    Args:
        - None

    Returns:
        - subprocess.Popen: The process running the server
    """
    # Fist we check that everything is installed
    downloadPokemonShowdown()

    # We start the server
    serverProcess = subprocess.Popen(
        ["bash", "startServer.sh"],
        stdout=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,  # Start the process in a new process group
    )

    # We wait for 'Test your server' to appear
    try:
        for line in serverProcess.stdout:
            print(line.strip(), flush=True)  # Print the output for debugging purposes
            if "Test your server" in line:
                print("Found the target message: 'Test your server'", flush=True)
                # It is ready
                time.sleep(1)  # Give it an extra second to ensure it's fully ready
                break
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
    finally:
        atexit.register(endProcess, serverProcess)

    return serverProcess


def downloadPokemonShowdown() -> None:
    """
    Downloads the Pokemon Showdown server.

    Args:
        - None

    Returns:
        - None
    """
    try:
        result = subprocess.run(
            ["bash", "downloadServer.sh"], capture_output=True, text=True, check=True
        )
        print(result.stdout.strip(), flush=True)
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr)


def getServerConfiguration() -> ServerConfiguration:
    """
    Returns a ServerConfiguration instance with the appropriate URLs.

    Args:
        - None

    Returns:
        - ServerConfiguration: An instance of ServerConfiguration with the appropriate URLs.
    """
    return ServerConfiguration(
        f"ws://localhost:{int(os.getenv("SERVER_PORT"))}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
