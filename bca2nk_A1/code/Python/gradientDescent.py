from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv

class gradientDescent:
    #GIVEN FUNCTIONS ——————————————————————————————————————————————————
    def gaussian_noise(self,img_gray,sigma): #Applies noise to an image
        row,col= img_gray.shape
        mean = 0
        var = sigma
        sigma = var ** 0.3
        gaussian = np.random.normal(mean, sigma, (row, col)) 
        noisy_img = img_gray + gaussian
        return noisy_img

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

    # CREATED FUNCTIONS —————————————————————————————————————————————————————————————
    def magnitude(self,matrix): #Determines the magnitude of a 2D matrix
        return np.linalg.norm(matrix,ord=2)

    def gradient(self, u, img, LAMBDA, EPSILON): #Energy gradient based on input u (u = k)
        return (-2 * LAMBDA *(img - u)-self.div(img, EPSILON))
    
    def dx(self, x): #Determines fowardX at each pixel, shortens name
        return self.forward_difference_x(x) 
    
    def dy(self, y): #Determines fowardY differences at each pixel, shortens name
        return self.forward_difference_y(y)

    def du(self, u,epsilon): #Denominator used in divergence function (div)
         return np.sqrt(self.dx(u)**2 + self.dy(u)**2)+epsilon 

    def div(self, u, epsilon): #Detemines the divergence of u (part of energy gradient function); uses a minor shift of epsilon
        return self.dx(np.divide(self.dx(u),self.du(u,epsilon)))+self.dy((np.divide(self.dy(u),self.du(u, epsilon))))


    def doGradientDescent(self,sigma): #Performs gradient descent algorithm based on an image
        
        '''
        Pseudocode 

        initialize u0

        for (k = 1; k<iter; k++):
            u^(k+1) = u^(k) - (step size) * gradient of u 
            check (E(u^(k+1)) & E(u^(k)) -> compare until convergence)
            save value of (k,E(u)) - use plot function

        '''
        
        #CONSTS/INPUTS ———————————————————————————————————————————————————————————————————————————————
        ITERATIONS = 100 #Max iterations
        LAMBDA = 0.9 #Scalar
        ALPHA = 0.6 #Step-size/Learning Rate
        EPSILON = 0.00001 #Small shift value when computing divergence
        THRESHOLD = 0.1**16 #CONVERGENCE INTERVAL based on differences in curr and prev gradients
        SIGMA = sigma #Noise Threshold
        u0 = 0.2 #INITIAL U START
        

        img = plt.imread('lena.png')
        img_gray = rgb2gray(img)
        noised = self.gaussian_noise(img_gray,SIGMA)

        cv2.resize(noised, (0,0), fx = 0.5, fy = 0.5) #Downscale img

        convPlotX = np.array([])
        convPlotY = np.array([])

        u = deepcopy(noised) + u0
        
        fig, ax = plt.subplots(1,3)
        prevGradient = 0

        # GRADIENT DESCENT ALGORITHM —————————————————————————————————————————————————————————————
        for k in range(1,ITERATIONS):
            #print(k)
            gradient = self.gradient(u, noised, LAMBDA, EPSILON)
            u = u - ALPHA * gradient

            if ( k > 5 and abs(self.magnitude(gradient)-self.magnitude(prevGradient)) < THRESHOLD):
                break

            convPlotX = np.append(convPlotX, k)
            convPlotY = np.append(convPlotY, self.magnitude(gradient))

            prevGradient = gradient.copy()

        #DISPLAY RESULTS ———————————————————————————————————————————————————————————————————————————
        fig.suptitle("Output", fontsize=20)
        fig.set_size_inches((12, 8)) #Increases window size, remove line if too big
        
        ax[0].imshow(noised, cmap ="gray") #Display noised image
        ax0Title = "Grayscale with Noise of " + str(SIGMA)
        ax[0].set_title(ax0Title, fontsize=8)
        ax[1].imshow(u, cmap ="gray") #Display denoised image
        ax[1].set_title("Grayscale Denoised", fontsize=8)
        ax[2].plot(convPlotX, convPlotY)
        ax[2].set_title("Energy Plot", fontsize=8) #Display energy plot
        ax[2].set_xlabel("k (Number of Iterations) ")
        ax[2].set_ylabel("E(u)")

        #Attempt to revert denoised image into color
        #uConv = u.astype(dtype=cv2.CV_32F)
        #rgb = cv2.cvtColor(uConv, cv2.COLOR_GRAY2BGR)
        #ax[2].imshow(rgb)
        
        plt.show()
    
#INPUTS————————————————————————————————————————————————————
obj = gradientDescent()
sigmaVals = [0.01,0.05,0.1]

for sig in sigmaVals:
    #Display initial image as well as image with noise
    img = plt.imread('lena.png')
    img_gray = rgb2gray(img)
    sigma = sig #Factor of noise

    noised = obj.gaussian_noise(img_gray, sigma)

    #Display input lena image and noised image
    fig, ax = plt.subplots(1,3)
    fig.suptitle("Input", fontsize=20)
    fig.set_size_inches((12, 8)) #Increases window size, remove line if too big
    
    ax[0].imshow(img, cmap ='gray')
    ax[0].set_title("Original", fontsize=8)
    ax[1].imshow(img_gray, cmap = 'gray')
    ax[1].set_title("Grayscale", fontsize=8)
    ax[2].imshow(noised, cmap = 'gray')
    ax2Title = "Grayscale with Noise of "+ str(sig) 
    ax[2].set_title(ax2Title, fontsize=8)
    plt.show()

    obj.doGradientDescent(sig)
    



    