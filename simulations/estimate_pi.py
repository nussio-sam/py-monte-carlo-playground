from random import random

def estim_pi(sample_count: int) -> int:
    pi_count = 0
    for i in range(sample_count):
        y_coord = random()
        x_coord = random()
        if x_coord**2 + y_coord**2 < 1:
            pi_count += 1
    return pi_count / sample_count * 4
