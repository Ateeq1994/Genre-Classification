"""
This script contains preprocessing functions.
"""
from PIL import Image

# Helper function to resize input images
def resize_image(src_img, size=(127, 127), bg_color="white"):
    """
    Resizes the images for training
    :param: images, image size (width, height), background color as string
    :return: resized image
    """

    # Rescale the image so the longest edge is the right size
    src_img.thumbnail(size, Image.ANTIALIAS)

    new_image = Image.new("RGB", size, bg_color)

    # Paste the rescaled image onto the new background
    new_image.paste(src_img, (int((size[0] - src_img.size[0]) / 2),
                              int((size[1] - src_img.size[1]) / 2)))
    return new_image
