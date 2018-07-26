import numpy as np
from skimage.filters import threshold_li

def calc_bounding_box(image):
    """
    Calculate bounding box of an object in an image

    @param image: Input image
    @return Extent of deformation in image (x_start, x_end, y_start, y_end)
    """

    thresh = threshold_li(image)
    thresh_image = np.where(image < thresh, 0, 1)
    column_maximums = np.max(thresh_image, axis=0)
    row_maximums = np.max(thresh_image, axis=1)
    x_start = np.argmax(column_maximums)
    x_end = len(column_maximums) - np.argmax(column_maximums[::-1]) - 1

    y_start = np.argmax(row_maximums)
    y_end = len(row_maximums) - np.argmax(row_maximums[::-1]) - 1

    return (x_start, x_end, y_start,  y_end)


def determine_deformation_bounding_box(deformations):
    """
    Determine bounds around a deformation

    @param deformations: Input deformations
    @return Bounding box large enough to include deformation in all directions (x_start, x_end, y_start, y_end)
    """
    bounds = np.stack([calc_bounding_box(np.abs(deformations[i,:,:])) for i in range(3)])
    return (np.min(bounds[:,0]), np.max(bounds[:,1]), np.min(bounds[:,2]), np.max(bounds[:,3]))



def determine_x_y_bounds(deformations, x_array, y_array, offset=5000):
    """
    Determine the x and y positions that bound a deformation

    @param deformations: Input deformations
    @param x_array: X coordinates
    @param y_array: Y coordinates
    @param offset: Extra padding around measured bounds
    @return Bounds in units of x_array and y_array with padding (x_start, x_end, y_start, y_end)
    """
    bounding_box = determine_deformation_bounding_box(deformations)
    x_start, x_end = x_array[0, bounding_box[:2]]
    y_start, y_end = y_array[bounding_box[2:], 0]

    return x_start-offset, x_end+offset, y_start-offset, y_end+offset
