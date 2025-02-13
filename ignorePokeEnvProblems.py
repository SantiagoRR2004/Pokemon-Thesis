"""
This is a class to ignore the error
messages that come from poke_env.

I am no able to ignore all the messages

To use this just import this file
"""

import logging


class IgnoreSpecificMessagesFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if "Unmanaged move message format received" in message:
            return False  # Return False to ignore this message

        return True


# Set up logging
logger = logging.getLogger()
handler = logging.StreamHandler()

# Add the custom filter to the handler
handler.addFilter(IgnoreSpecificMessagesFilter())

# Set the log level
logger.setLevel(logging.ERROR)
logger.addHandler(handler)
