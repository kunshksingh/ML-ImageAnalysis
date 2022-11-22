import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

class fourierTestform:
   
    def fourierTransform():
        #PART A —————————————————————————————————————————————————————————————
        lenaImage = plt.imread('lena.png')

        #Perform's fourier transform on lena image
        ftShift = np.fft.fftshift(np.fft.fft2(lenaImage, axes=(0,1)))
        ftShift2 = np.fft.fftshift(np.fft.fft2(lenaImage, axes=(0,1)))
        ftShift3 = np.fft.fftshift(np.fft.fft2(lenaImage, axes=(0,1))) 

        fig, ax = plt.subplots(1,2)
        fig.suptitle('PART A')
        fig. set_size_inches((12, 8)) #Increases window size, remove line if too big

        #Setup Figure/Subplots
        ax[0].imshow(lenaImage)
        ax[0].set_title("Original Image", fontsize=5)
        ax[1].imshow(np.log(np.abs(ftShift)))
        ax[1].set_title("Fourier Transform of Lena (With centered low frequencies)", fontsize=5)
        plt.show()
        
        #PART B —————————————————————————————————————————————————————————————
        mask40 = np.zeros_like(lenaImage)
        mask20 = np.zeros_like(lenaImage)
        mask10 = np.zeros_like(lenaImage)
        
        x = int(mask40.shape[1]/2) #CenterX of frquency domain
        y = int(mask40.shape[0]/2) #CenterY of frquency domain

        #Mask fourier transforms with rectangle to 0 out high frequencies beyond a threshold
        cv2.rectangle(mask40, ((x-40),(y-40)), ((x+40),(y+40)), (255,255,255), -1)[0]#Square mask for threshold=40^2
        cv2.rectangle(mask20, ((x-20),(y-20)), ((x+20),(y+20)), (255,255,255), -1)[0]#Square mask for threshold=20^2
        cv2.rectangle(mask10, ((x-10),(y-10)) , ((x+10),(y+10)), (255,255,255), -1)[0]#Square mask for threshold=10^2

        ftShift *= mask40/255
        ftShift2 *= mask20/255
        ftShift3 *= mask10/255
        
        #PART C —————————————————————————————————————————————————————————————

        #Reconstruct original image using inverse fourier transform
        reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(ftShift), axes=(0,1))) 
        reconstructed2 = np.abs(np.fft.ifft2(np.fft.ifftshift(ftShift2), axes=(0,1))) 
        reconstructed3 = np.abs(np.fft.ifft2(np.fft.ifftshift(ftShift3), axes=(0,1))) 
        

        #Setup Figure/Subplots
        fig, ax = plt.subplots(1,4)
        fig.suptitle('PART B & C')
        fig.set_size_inches((12, 8)) #Increases window size, remove line if too big

        ax[0].imshow(lenaImage)
        ax[0].set_title("Original Image", fontsize=5)
        ax[1].imshow(reconstructed)
        ax[1].set_title("Lena image removing frequencies beyond 40^2", fontsize=5)
        ax[2].imshow(reconstructed2)
        ax[2].set_title("Lena image removing frequencies beyond 20^2", fontsize=5)
        ax[3].imshow(reconstructed3)
        ax[3].set_title("Lena image removing frequencies beyond 10^2", fontsize=5)
        plt.show()
        

        '''
        CODE: DISPLAY LENA IMAGE WITH A SLIGHTLY VIGNETTE FILTER (BORDER IS ROUNDER/IRREGULAR/TINTED PURPLE) - Discovered while messing with numbers and order of ifftshift/ifft2

        vignette = np.log(np.abs(np.fft.ifftshift(np.fft.ifft2(ftShift))-0.9))
        orig = vignette.copy()
        vignette.transpose()
        vignette *= orig

        reconstructed += vignette#-np.abs(np.log(ftShift)/50)

        plt.imshow(reconstructed) 

        '''
    

    fourierTransform()
        