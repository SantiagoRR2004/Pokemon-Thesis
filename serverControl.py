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
    server_command = [
        "node",
        "pokemon-showdown/pokemon-showdown",
        "start",
        "--no-security",
    ]
    serverProcess = subprocess.Popen(server_command)

    atexit.register(endProcess, serverProcess)
