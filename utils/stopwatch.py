import time
from loguru import logger

class Stopwatch:
    """A simple stopwatch class to measure elapsed time."""
    
    def __init__(self):
        self.func_dict = {}
        self.start_time = None
        self.end_time = None

    def time(self, func, *args, **kwargs):
        """Decorator to time a function."""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        if func.__name__ not in self.func_dict:
            self.func_dict[func.__name__] = 0
        self.func_dict[func.__name__] += elapsed
        return result

    def __str__(self):
        """String representation of the stopwatch."""
        rep = f"Stopwatch:\n"
        for func_name, elapsed in self.func_dict.items():
            rep += f"Function '{func_name}' executed in {elapsed:.4f} seconds."
        return rep
    
    def __del__(self):
        """Destructor to log the elapsed times when the object is deleted."""
        logger.info(self.__str__())


stopwatch = Stopwatch() 

def timethis(func):
    """Decorator to time a function."""
    def wrapper(*args, **kwargs):
        result = stopwatch.time(func, *args, **kwargs)
        return result
    return wrapper