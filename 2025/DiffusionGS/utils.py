import trimesh 
import numpy as np 
def center_scale_mesh(mesh: trimesh.Trimesh):

  # Calculate the minimum, maximum bound and center
  min_bound = mesh.vertices.min(axis=0)
  max_bound = mesh.vertices.max(axis=0)
  center = (min_bound + max_bound) / 2
  print("Mininum bound of Original: ", min_bound)
  print("Maximum bound of Original: ", max_bound)
  print("Center: ", center)

  # Move all the vertices to center
  mesh.vertices -= center

  bound_size = max_bound - min_bound
  max_extent = np.max(bound_size)

  mesh.vertices /= (max_extent / 2)

  return mesh
