import SimpleITK as sitk
import numpy as np

def load_dicom_series_to_3d_image(dicom_folder):
    """
    Load a folder of 2D DICOM slices into a 3D SimpleITK image.
    
    Args:
        dicom_folder (str): Path to the folder containing DICOM slices.
    
    Returns:
        SimpleITK.Image: 3D image composed from the series.
    """
    # Initialize reader
    reader = sitk.ImageSeriesReader()

    # Get list of DICOM file names (sorted in correct order)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM files found in folder: {dicom_folder}")
    
    reader.SetFileNames(dicom_names)

    # Read the series
    image_3d = reader.Execute()

    print(f"[INFO] Loaded DICOM series with size: {image_3d.GetSize()}")
    return image_3d

# Load image and resample mask using SimpleITK 
def convert_np_to_sitk(pixel_array, header):
    sitk_image = sitk.GetImageFromArray(pixel_array.astype(np.uint8))
    sitk_image.SetSpacing([1,1,1])
    sitk_image.SetOrigin(header['space origin'])
    sitk_image.SetDirection(np.reshape(header['space directions'], -1))
    return sitk_image

def reconstruct_mask_in_image_space(mask_sitk, image_sitk):
    """
    Reconstruct the segmentation mask in the original image space.

    Parameters:
    - mask_sitk: SimpleITK image (cropped mask)
    - image_sitk: SimpleITK image (original image)

    Returns:
    - full_mask: a NumPy array with the shape of image_sitk and the mask embedded in the correct location
    """
    # Get index offset of mask origin in image space
    mask_start_index = image_sitk.TransformPhysicalPointToIndex(mask_sitk.GetOrigin())
    print(f"[INFO] Loaded mask with size: {mask_sitk.GetSize()}")

    # Convert mask to NumPy array
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    dx, dy, dz = mask_sitk.GetSize()

    # Prepare full-size array
    image_shape = image_sitk.GetSize()
    full_mask = np.zeros(image_shape, dtype=mask_array.dtype)

    # Compute insertion coordinates
    x, y, z = mask_start_index

    # Compute slice bounds and clip if necessary
    z_end = min(z + dz, image_shape[2])
    y_end = min(y + dy, image_shape[1])
    x_end = min(x + dx, image_shape[0])

    # Corresponding bounds in the mask
    mask_z_end = z_end - z
    mask_y_end = y_end - y
    mask_x_end = x_end - x

    full_mask[x:x_end, y:y_end, z:z_end] = mask_array.transpose(2,1,0)[:mask_x_end, :mask_y_end, :mask_z_end]

    # From np to SITK
    full_mask_sitk = sitk.GetImageFromArray(full_mask.transpose(2,1,0))
    full_mask_sitk.SetSpacing(image_sitk.GetSpacing())
    full_mask_sitk.SetOrigin(image_sitk.GetOrigin())
    full_mask_sitk.SetDirection(image_sitk.GetDirection())

    return full_mask_sitk