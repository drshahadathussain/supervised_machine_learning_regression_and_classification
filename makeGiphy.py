import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Create data
x = np.linspace(-5, 5, 100)  # w values
y = np.linspace(-5, 5, 100)  # b values
X, Y = np.meshgrid(x, y)

# Example cost function J(w, b) = w² + b²
Z = X**2 + Y**2

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with color gradient
surf = ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.8)

# Set labels
ax.set_xlabel('w', fontsize=12)
ax.set_ylabel('b', fontsize=12)
ax.set_zlabel('J(w, b)', fontsize=12)
ax.set_title('Cost Function Visualization', fontsize=14)

# Add colorbar
fig.colorbar(surf, shrink=0.5, aspect=10, label='J(w, b) value')

# Animation function
def update(frame):
    ax.view_init(elev=20, azim=frame)
    return fig,

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

# Save as GIF
ani.save('cost_function_rotation.gif', writer='pillow', fps=20, dpi=100)
plt.show()