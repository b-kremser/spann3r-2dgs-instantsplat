import json
import os
import shutil
from pathlib import Path

def prepare_scene(scene_directory, output_directory, no_images, first_image, last_image):
    """
    Prepares data for a single scene.

    Parameters:
        scene_directory (str): Path to the scene's directory.
        output_directory (str): Path to store the prepared data for this scene.
        no_images (int): Number of images to sample evenly.
        first_image (str): Name of the first image to include.
        last_image (str): Name of the last image to include.
    """
    # Convert input paths to Path objects for convenience
    scene_directory = Path(scene_directory)
    output_directory = Path(output_directory)

    # Locate the images and mesh file in the scene directory
    images_dir = scene_directory / "dslr" / "resized_images"
    mesh_file = scene_directory / "scans" / "mesh_aligned_0.05.ply"

    if not images_dir.exists() or not mesh_file.exists():
        raise FileNotFoundError(f"Missing images or mesh file in {scene_directory}")

    # Get all image files in the images directory
    image_files = sorted(images_dir.glob("*.JPG"))

    # Ensure first and last images exist in the directory
    if first_image not in [img.name for img in image_files] or last_image not in [img.name for img in image_files]:
        raise ValueError("Specified first_image or last_image does not exist in the images directory")

    # Filter images between first_image and last_image, inclusive
    start_index = next(i for i, img in enumerate(image_files) if img.name == first_image)
    end_index = next(i for i, img in enumerate(image_files) if img.name == last_image)
    filtered_images = image_files[start_index:end_index + 1]

    if len(filtered_images) < no_images:
        raise ValueError("Not enough images between first_image and last_image to sample")

    # Select evenly spaced samples
    step = len(filtered_images) / (no_images - 1)
    sampled_images = [filtered_images[round(i * step)] for i in range(no_images - 1)] + [filtered_images[-1]]

    # Create the output directories
    scene_output_dir = output_directory / scene_directory.name
    views_dir = scene_output_dir / f"{no_images}_views"
    images_output_dir = views_dir / "images"
    mesh_output_dir = views_dir / "mesh"

    images_output_dir.mkdir(parents=True, exist_ok=True)
    mesh_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy sampled images to the images output directory
    for i, image_file in enumerate(sampled_images):
        shutil.copy(image_file, images_output_dir / f"image_{i:04d}.jpg")

    # Copy the mesh file to the mesh output directory
    shutil.copy(mesh_file, mesh_output_dir / "mesh_aligned_0.05.ply")

    print(f"Finished processing scene: {scene_directory.name}")

if __name__ == "__main__":
    config_file = "prepare_scannet_data_config.json"

    with open(config_file, "r") as f:
        config = json.load(f)

    source_dir = Path(config["source_directory"])
    output_dir = Path(config["output_directory"])
    no_images = config["no_images"]

    for scene in config["scenes"]:
        scene_directory = Path.joinpath(source_dir, scene["scene_name"])

        prepare_scene(
            scene_directory=scene_directory,
            output_directory=output_dir,
            no_images=no_images,
            first_image=scene["first_image"],
            last_image=scene["last_image"],
        )
