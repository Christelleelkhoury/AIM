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

def resample_mask_to_image(mask, image, interpolator=sitk.sitkNearestNeighbor):
    """Resample the segmentation mask to match the reference image (e.g., T1w)"""
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    resample.SetInterpolator(interpolator)
    resample.SetTransform(sitk.Transform())  # Identity transform
    return resample.Execute(mask)