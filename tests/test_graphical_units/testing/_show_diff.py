import numpy as np


def show_diff_helper(
    frame_number: int,
    frame_data: np.ndarray,
    expected_frame_data: np.ndarray,
):
    """Will visually display with matplotlib differences between frame generated and the one expected."""
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    fig.suptitle(f"Test difference summary at frame {frame_number}", fontsize=16)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(frame_data)
    ax.set_title("Generated :")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(expected_frame_data)
    ax.set_title("Expected :")

    ax = fig.add_subplot(gs[1, :])
    diff_im = expected_frame_data.copy()
    diff_im = np.where(
        frame_data != np.array([0, 0, 0, 255]),
        np.array([0, 255, 0, 255], dtype="uint8"),
        np.array([0, 0, 0, 255], dtype="uint8"),
    )  # Set any non-black pixels to green
    np.putmask(
        diff_im,
        expected_frame_data != frame_data,
        np.array([255, 0, 0, 255], dtype="uint8"),
    )  # Set any different pixels to red
    ax.imshow(diff_im, interpolation="nearest")
    ax.set_title("Differences summary : (green = same, red = different)")

    plt.show()
