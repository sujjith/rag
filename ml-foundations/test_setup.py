import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Test numpy
arr = np.array([1, 2, 3, 4, 5])
print(f"✅ NumPy array: {arr}")
# Test pandas
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(f"✅ Pandas DataFrame:\n{df}")
# Test matplotlib (create simple plot)
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig('test_plot.png')
print("✅ Matplotlib plot saved as test_plot.png")
