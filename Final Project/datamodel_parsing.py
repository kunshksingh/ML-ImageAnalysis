#Convert a directory of 2D .mhd files to .png
# Import the necessary modules
import os
import SimpleITK as sitk
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# Specify the path to the folder containing the .mhd files
testing = "Classification_data/Testing1/Diseased"
training = "Classification_data/Training/Diseased"

# Create an ImageSeriesReader object to read the .mhd files
reader = sitk.ImageSeriesReader()

# Get a list of the file names in the folder
file_names = os.listdir(training)

# Loop through the file names
for file_name in file_names:
    # Check if the file is a .mhd file
    if file_name.endswith(".mdh"):
        # Get the file path
        file_path = os.path.join(training, file_name)

        # Read the image
        # image = sitk.ReadImage(file_path)
        # imageArr = sitk.GetArrayFromImage(image)
        # print(imageArr.shape)

        reader = sitk.ImageFileReader()
        #reader.SetImageIO()
        reader.SetFileName(file_path)
        image = reader.Execute()

        imageArr = sitk.GetArrayFromImage(image)

        writer = sitk.ImageFileWriter()
        writer.SetImageIO("PNGImageIO")
        writer.SetFileName(file_name)
        writer.Execute(imageArr)
        
        # rawData = open(file_name, 'rb').read()
        # imgSize = (256,256)
        # print(rawData)
        # img = Image.frombytes('L',imgSize,rawData)
        # img.save(file_name[:-4]+".png")
        # plt.imshow(img)
        # plt.show()
        





        #im = Image.fromarray(imageArr).convert('L')

        #im.save(file_path[:-4]+".png")
        # print(file_path[:-4]+".png")
        
