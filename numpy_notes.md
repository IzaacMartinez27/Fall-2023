# NumPy and Matplotlib Functions README

This README provides a brief explanation of some commonly used NumPy and Matplotlib functions for data manipulation and visualization in Python.

## NumPy Functions

### `np.zeros(shape)`

- Creates a NumPy array filled with zeros.
- `shape`: A tuple specifying the dimensions of the array (e.g., `(rows, columns)`).

Example:
```python
import numpy as np
zeros_array = np.zeros((3, 4))
```

np.ones(shape)


```
import numpy as np
ones_array = np.ones((2, 3))
```

np.eye(N)


```
import numpy as np
identity_matrix = np.eye(3)
```

np.linspace(start, stop, num)

```
import numpy as np
evenly_spaced = np.linspace(0, 1, 5)
```

plt.imshow(image)

```
import matplotlib.pyplot as plt
plt.imshow(image_array, cmap='gray')
plt.title('Sample Image')
plt.show()
```

plt.plot

```import matplotlib.pyplot as plt
plt.plot(x_values, y_values, marker='o', linestyle='--', color='b', label='Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')
plt.legend()
plt.show()
```

```
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 3, 7]

# Plot with different marker types
plt.plot(x, y, marker='o', label='Circle')
plt.plot(x, y, marker='s', label='Square')
plt.plot(x, y, marker='*', label='Star')

plt.legend()
plt.show()
```








