# Generated using chatGPT
import argparse
from PIL import Image
import os


def process_image(image_path, velocity=False, normal=False):
    """
    Flips image horizontally.
    Optionally inverts red channel and adds 127 (clipped to 255).
    Saves image in the same folder with '_fliped' appended.
    """
    image = Image.open(image_path).convert("RGB")

    # Flip horizontally
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

    if velocity:
        pixels = flipped.load()
        width, height = flipped.size

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]

                if g < 100:
                    continue

                # Invert red
                r = 255 - r

                # Add 127 and clip
                r = min(255, r + 127)

                pixels[x, y] = (r, g, b)

    elif normal:
        pixels = flipped.load()
        width, height = flipped.size

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                rf = float(r) / 255
                gf = float(g) / 255
                bf = float(b) / 255
                if rf * rf + gf * gf + bf * bf < 0.9:
                    continue

                # Invert red
                r = 255 - r

                pixels[x, y] = (r, g, b)

    # Build output filename
    directory, filename = os.path.split(image_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_fliped{ext}"
    output_path = os.path.join(directory, output_filename)

    flipped.save(output_path)


def process_folder(directory, flip_x_vel=False, flip_x_normal=False):
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(directory, filename)

            # Avoid re-processing already processed files
            if "_fliped" not in filename:
                process_image(input_path, flip_x_vel, flip_x_normal)

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flip images horizontally and optionally modify red channel."
    )
    parser.add_argument("directory", type=str, help="Path to image directory")
    parser.add_argument(
        "--color",
        action="store_true",
        help="Enable red channel inversion and modification (default: False)",
    )
    parser.add_argument(
        "--normal",
        action="store_true",
        help="Enable red channel inversion and modification (default: False)",
    )

    args = parser.parse_args()

    process_folder(args.directory, args.color, args.normal)
