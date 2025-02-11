from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal
from typing import Dict
import plotly.graph_objects as go


def create_3D_Gaussians(points, colors, grid_size:int=100) -> Dict[str, np.array]:
  """ Create a set of 3D Gausssians.

  Args:
    sfm_path: Path to the SfM folder containing the COLMAP outputs.
  """

  points, colors, _ = create_SfM_points(sfm_path)

  # Calculate the mean
  mu = np.mean(points, axis=0)

  # Calculate the covariance matrix
  covariance_matrix = np.cov(points, rowvar=False)

  # Define the grid range
  x_range = np.linspace(points[:, 0].min(), points[:, 0].max(), grid_size)
  y_range = np.linspace(points[:, 1].min(), points[:, 1].max(), grid_size)
  z_range = np.linspace(points[:, 2].min(), points[:, 2].max(), grid_size)

  # Create a 3D grid
  x, y, z = np.meshgrid(x_range, y_range, z_range)
  grid_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

  # Create a multivariate Gaussian distribution
  gaussian = multivariate_normal(mean=mu, cov=covariance_matrix)

  # Reshape PDF values to match the grid
  pdf_values = gaussian.pdf(grid_points).reshape(x.shape)

  return  {
      "pdf_values": pdf_values,
      "x_range": x_range,
      "y_range": y_range,
      "z_range": z_range
  }

def visualize_3D_Gaussians(data: Dict[str, np.array]):

  pdf_values = data["pdf_values"]
  x_range = data["x_range"]
  y_range = data["y_range"]
  z_range = data["z_range"]

  # Generate Plotly Volume plot
  fig = go.Figure(data=go.Volume(
      x=np.repeat(x_range, len(y_range) * len(z_range)),
      y=np.tile(np.repeat(y_range, len(z_range)), len(x_range)),
      z=np.tile(z_range, len(x_range) * len(y_range)),
      value=pdf_values.ravel(),
      isomin=pdf_values.min(),
      isomax=pdf_values.max(),
      opacity=0.1,
      surface_count=10,
      colorscale="Viridis"
  ))

  # Setup plot
  fig.update_layout(
      scene=dict(
          xaxis_title="X",
          yaxis_title="Y",
          zaxis_title="Z",
          aspectmode="cube"
      ),
      title="3D Gaussian Visualization"
  )

  fig.show()
