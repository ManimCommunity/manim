import os
import sys

import numpy as np
from PIL import Image


def main():
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)
    npz_file = sys.argv[1]
    output_folder = sys.argv[2]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = np.load(npz_file)
    if "frame_data" not in data:
        print("The given file did not have frame_data.")
        print("Are you sure this is from a Manim Graphical Unit Test?")
        sys.exit(2)
    frames = data["frame_data"]
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(os.path.join(output_folder, f"frame{i}.png"))
    print(f"Saved {len(frames)} frames to {output_folder}")


def print_usage():
    print("Manim Graphical Test Frame Extractor")
    print(
        "This tool outputs the frames of a Graphical Unit Test "
        "stored within a .npz file, typically found under "
        r"//tests/test_graphical_units/control_data"
    )
    print()
    print("usage:")
    print("python3 extract_frames.py npz_file output_directory")


if __name__ == "__main__":
    main()
