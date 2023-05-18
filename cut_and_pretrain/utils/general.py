import numpy as np
import json


def get_three_points(x_step, y_step, size):
    top_left = (int(x_step * size), int(y_step * size))
    bottom_right = (int(top_left[0] + size), int(top_left[1] + size))
    center = (int((top_left[0] + bottom_right[0]) // 2), int((top_left[1] + bottom_right[1]) // 2))
    return top_left, bottom_right, center


def downsample_image(slide, downsampling_factor=16, mode="numpy"):
    best_downsampling_level = slide.get_best_level_for_downsample(downsampling_factor + 0.1)
#     best_downsampling_level = 3
    # Get the image at the requested scale
    svs_native_levelimg = slide.read_region((0, 0), best_downsampling_level,
                                            slide.level_dimensions[best_downsampling_level])
    
    target_size = tuple([int(x // downsampling_factor) for x in slide.dimensions])
    img = svs_native_levelimg.resize(target_size)

    # By default, return a numpy array as RGB, otherwise, return PIL image
    if mode == "numpy":
        # Remove the alpha channel
        img = np.array(img.convert("RGB"))

    return img, best_downsampling_level


def keep_patch(mask_patch, thres, bg_color):
    """Specialized selector for otsu or adaptive TileGenerator.

    Determines if a mask tile contains a certain percentage of foreground.

    Args:
        mask_patch: Numpy array for the current mask tile.
        thres: Float indicating the minimum foreground content [0, 1] in
            the patch to select the tile.
        bg_color: Numpy array with the background color for the mask.

    Returns:
        _: Integer [0/1] indicating if the tile has been selected or not.
    """
    # print(mask_patch.shape)
    # print(bg_color.shape)
    bg = np.all(mask_patch == bg_color, axis=2)
    # print(bg.shape)
    bg_proportion = np.sum(bg) / bg.size

    if bg_proportion <= (1 - thres):
        output = 1
    else:
        output = 0

    return output


def out_of_bound(w, h, x, y):
    return x >= w or y >= h


def dump_json(data_dict, filename):
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(data_dict, fp)


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        data_dict = json.load(fp)
    return data_dict
