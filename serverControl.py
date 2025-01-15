import subprocess
import signal
import atexit


def endProcess(process: subprocess.Popen) -> None:
    """
    Ends the given process.

    Args:
        - process (subprocess.Popen): The process to end

    Returns:
        - None
    """
    # Step 1: Send a SIGINT (interrupt) signal to the process (similar to pressing Ctrl+C)
    process.send_signal(signal.SIGINT)

    # Step 2: Wait for the process to finish
    process.wait()

    print("Server shut down successfully.")


def startServer() -> None:
    """
    Starts the Pokemon Showdown server.

    Args:
        - None

    Returns:
        - None
    """
    # Fist we check that everything is installed
    downloadPokemonShowdown()

    # We start the server
    server_command = [
        "node",
        "pokemon-showdown/pokemon-showdown",
        "start",
        "--no-security",
    ]
    serverProcess = subprocess.Popen(server_command)

    atexit.register(endProcess, serverProcess)


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
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr)
