import time
import warnings

class RateLimiter:
    """
    A simple rate limiter to keep a loop running at a fixed frequency.
    """
    def __init__(self, frequency, warn=True):
        """
        Initialize the RateLimiter.
        
        Args:
            frequency (float): The target frequency in Hz.
            warn (bool): Whether to issue a warning if the loop runs slower than target.
        """
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.warn = warn
        self.next_tick = time.time()

    def sleep(self):
        """
        Sleeps for the remainder of the period.
        """
        # Update the target time for the next tick
        self.next_tick += self.period
        now = time.time()
        
        sleep_duration = self.next_tick - now
        
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        else:
            # We are behind schedule
            if self.warn:
                warnings.warn(f"RateLimiter: Loop running slow. Behind by {-sleep_duration:.4f} s")
            
            # If we are behind, reset the schedule to now to avoid bursty catch-up behavior
            self.next_tick = now
