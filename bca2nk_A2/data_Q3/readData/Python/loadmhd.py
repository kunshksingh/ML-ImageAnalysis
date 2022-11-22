import SimpleITK as sitk

PATHOFVELOCITY = "code+data_Q3/data/v0Spatial.mhd"
PATHOFSOURCE = "code+data_Q3/data/v0Spatial.mhd"

'''Read in data as a 1, 100, 100, 3 vector field, please change the PATHOFVELOCITY to the directory path of velocity ''' 
velocity = sitk.GetArrayFromImage(sitk.ReadImage(PATHOFVELOCITY))

'''Read in data as a 1, 100, 100  image, please change the PATHOFSOURCE to the directory path of source '''
source= sitk.GetArrayFromImage(sitk.ReadImage(PATHOFSOURCE))