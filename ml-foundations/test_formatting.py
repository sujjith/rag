import pandas as pd
import numpy as np


def badly_formatted(x, y, z):
    result = x + y + z
    return result


data = [1, 2, 3, 4, 5]
print(badly_formatted(1, 2, 3))
