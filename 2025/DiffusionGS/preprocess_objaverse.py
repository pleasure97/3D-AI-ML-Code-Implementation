from utils import center_scale_mesh 
import trimesh
import numpy as np
import os 

def sample_random_views(num_images=32, radius=1.):
  camera_positions = []
  view_directions = []

  for _ in range(num_images):
      theta = np.random.uniform(-2, 2)
      phi = np.random.uniform(-1, 1.5)

      cam_x = radius * np.sin(phi) * np.cos(theta)
      cam_y = radius * np.sin(phi) * np.sin(theta)
      cam_z = radius * np.cos(phi)

      position = np.array([cam_x, cam_y, cam_z])
      direction = -position / np.linalg.norm(position)

      camera_positions.append(position)
      view_directions.append(direction)

  return camera_positions, view_directions


def render_images_trimesh(mesh_file, output_dir, num_images=32, fov=50, image_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)

    mesh = trimesh.load(mesh_file, force='mesh')
    mesh = normalize_mesh(mesh)

    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    camera_positions, view_directions = sample_random_views(num_images)

    for i, (position, direction) in enumerate(zip(camera_positions, view_directions)):
        try:
            camera = trimesh.scene.Camera(fov=(fov, fov), resolution=image_size)
            scene.camera = camera

            look_at_matrix = trimesh.scene.cameras.look_at(points=[position], fov=fov)

            scene.camera_transform = look_at_matrix

            image_data = scene.save_image(resolution=image_size)
            if image_data is not None:
                image_path = os.path.join(output_dir, f"{i:03d}.png")
                with open(image_path, 'wb') as f:
                    f.write(image_data)
        except ZeroDivisionError as e:
            print(f"[ERROR] ZeroDivisionError for view {i} : {e}")
            continue

    print(f"Saved {num_images} images to {output_dir}")
    print(f"Saved {num_images} images to {output_dir}")
