import math
import SimpleITK as sitk
import numpy as np
import scipy as sp
from scipy.ndimage import map_coordinates as mc
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray


class Solution:

    #PATH TO INPUTS ————————————————————————————————————————————————————
    PATHOFVELOCITY = "data_Q3/data/initialV/v0Spatial.mhd"
    PATHOFSOURCE = "data_Q3/data/sourceImage/source.mhd"
    

    #GIVEN FUNCTIONS ———————————————————————————————————————————————————
    def velocity(self): #Gets velocity based on input image
        return sitk.GetArrayFromImage(sitk.ReadImage(self.PATHOFVELOCITY))

    def src(self): #Gets src based on input image
        return sitk.GetArrayFromImage(sitk.ReadImage(self.PATHOFSOURCE))

    #FUNCTIONS FROM PS1 —————————————————————————————————————————————————
    def forward_difference_x(self,image): #Computes the x component of the gradient
        rows, cols = image.shape
        d = np.zeros((rows,cols))
        d[:,1:cols-1] = image[:,1:cols-1] - image[:,0:cols-2]
        d[:,0] = image[:,0] - image[:,cols-1]
        return d

    def forward_difference_y(self,image): #Computes the y component of the gradient
        rows, cols = image.shape
        d = np.zeros((rows,cols))
        d[1:rows-1, :] = image[1:rows-1, :] - image[0:rows-2, :]
        d[0,:] = image[0,:] - image[rows-1,:]
        return d
    def dx(self, x): #Determines fowardX at each pixel, shortens name
        return self.forward_difference_x(x) 
    
    def dy(self, y): #Determines fowardY differences at each pixel, shortens name
        return self.forward_difference_y(y)
    
    #NEW DIVERGENCE FUNCTIONS —————————————————————————————————————————————
    def divx(self, vx,vy, epsilon=0.000001): #Detemines the divergence of vx term; uses a minor shift of epsilon
        return self.dx(vx*vx+epsilon)+self.dx(vx*vy+epsilon)
    def divy(self, vx,vy, epsilon=0.000001): #Detemines the divergence of vy term; uses a minor shift of epsilon
        return self.dy(vy*vx+epsilon)+self.dy(vy*vy+epsilon)
    
    
    def smoothing(self, u): #Smooths image (from pset1)
        #Perform's fourier transform on input image
        ftShift = np.fft.fftshift(np.fft.fft2(u, axes=(0,1)))

        mask16 = np.zeros_like(u)
        x = int(mask16.shape[1]/2) #CenterX of frquency domain
        y = int(mask16.shape[0]/2) #CenterY of frquency domain

        #Mask fourier transform with rectangle to 0 out high frequencies beyond 16^2
        cv2.rectangle(mask16, ((x-16),(y-16)), ((x+16),(y+16)), (255,255,255), -1)[0]#Square mask for threshold=16^2
        ftShift *= mask16/255 #Apply mask

        #Reconstruct our fourier transform from frequency domain
        u = np.abs(np.fft.ifft2(np.fft.ifftshift(ftShift), axes=(0,1))) 
        
        return u

    #MAIN FUNC ——————————————————————————————————————————————————————————————————
    def geodesicShooter(self): #Performs Geodesic Shooting for Diffeomorphic Image Registration

        #CONSTS/INPUTS
        STEPS = 2

        src = np.squeeze((self.src())) #Get source image
        src2 = plt.imread('lena.png') #Testing lena too!
        src3 = plt.imread('graph.jpeg') #And a standard cartesian graph

        src2 = rgb2gray(src2)
        src3 = rgb2gray(src3)
        src2 = cv2.resize(src2, (0,0), fx = 0.1953125, fy = 0.1953125) #Downsize Lena to 100x100
        src3 = cv2.resize(src3, (0,0), fx = 0.3125, fy = 0.31347962382) #Downsize graph to 100x100
    
        input = self.velocity() #Get input velocities
        input = np.squeeze(input)

        vx =  input[:,:,0] #Splice input matrix to get the x component of velocity
        vy =  input[:,:,1] #Splice input matrix to get the y component of velocity
        

        phi_x, phi_y  = np.mgrid[0:100, 0:100].astype(float) #Initialize phi with mesh grid


        #Perform Geodesic Shooting in combination with Eulers to approximate t = 1
        for i in range(STEPS):
            #Find Jacobian components of velocity matrix
            vxx = self.dx(vx)
            vxy = self.dy(vx)
            vyx = self.dx(vy)
            vyy = self.dy(vy)
         
            #Transpose Jacobian components
            jacobian_transpose = np.transpose([[vxx, vxy],[vyx, vyy]])
            vxx = jacobian_transpose[:,:,0,0]
            vxy = jacobian_transpose[:,:,0,1]
            vyx = jacobian_transpose[:,:,1,0]
            vyy = jacobian_transpose[:,:,1,1]

            #Compute Jacobian term multiplied with velocity matrix
            jacobian_x = vxx*vx + vxy*vy
            jacobian_y = vyx*vx + vyy*vy
           
            #Perform Eulers with smoothing operator
            dvx_dt = -1*self.smoothing(jacobian_x + self.divx(vx,vy)) #dvx/dt
            dvy_dt = -1*self.smoothing(jacobian_y + self.divy(vx,vy)) #dvy/dt
            vx += dvx_dt * 1/STEPS
            vy += dvy_dt * 1/STEPS

            #Perform interpolation to find dphi_dt
            dphi_x_dt = mc(vx, (phi_x, phi_y), order=3)
            dphi_y_dt = mc(vy, (phi_x,phi_y), order=3)
            
            phi_x += dphi_x_dt * 1/STEPS
            phi_y += dphi_y_dt * 1/STEPS

        
        #DISPLAY RESULTS ——————————————————————————————————————————————————————————————
        fig, ax = plt.subplots(3,2)
        fig.set_size_inches((12, 8)) #Increases window size, remove line if too big
        plt.subplots_adjust(left=0.3, bottom=0.1, right=0.7,top=0.9,wspace=0.4,hspace=0.4)

        #Display initial conditions
        ax[0][0].imshow(src, cmap = 'plasma')
        ax[0][0].set_title("Given Image", fontsize=9)
        ax[1][0].imshow(src2, cmap='gray')
        ax[1][0].set_title("Source Lena", fontsize=9)
        ax[2][0].imshow(src3, cmap='gray')
        ax[2][0].set_title("Source Graph", fontsize=9)

        #Interpolate source images by phi
        dst = mc(src, (phi_x,phi_y), order=3)
        dst2 = mc(src2, (phi_x,phi_y), order=3)
        dst3 = mc(src3, (phi_x,phi_y), order=3)

        #Show Results
        ax[0][1].imshow(dst, cmap='plasma')
        ax[0][1].set_title("Deformed Given Image", fontsize=9)
        ax[1][1].imshow(dst2, cmap='gray')
        ax[1][1].set_title("Deformed Lena", fontsize=9)
        ax[2][1].imshow(dst3, cmap='gray')
        ax[2][1].set_title("Deformed Graph", fontsize=9)
        plt.show()

sol = Solution()
sol.geodesicShooter()
    
