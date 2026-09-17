import time


def start_timer():
    """
    Start a high-resolution timer.
    """
    return time.perf_counter()


def stop_timer(start_time):
    """
    Calculate elapsed time in seconds.
    """
    return time.perf_counter() - start_time


