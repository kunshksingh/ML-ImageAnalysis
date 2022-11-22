import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cv2
from scipy.io import loadmat

class Solution:
    #FUNCTIONS FROM PS1 —————————————————————————————————————————————————
    def magnitude(matrix): #Determines the magnitude of a 2D matrix
            return np.linalg.norm(matrix,ord=2)
    
    #NEW AUGMENT FUNCTION ——————————————————————————————————————————————
    def augment(img):
        h, w = img.shape[:2] #Height/width of lena
        random_theta = random.uniform(0,60)

        random_t = random.uniform(-5,5)
        #Apply the given augmentation to the image
        #ROTATION
        rotation = np.array([[math.cos(math.radians(random_theta)), -1*math.sin(math.radians(random_theta)),1],[math.sin(math.radians(random_theta)),math.cos(math.radians(random_theta)),1]]) #Rotation Matrix based on random_theta
        img = cv2.warpAffine(img, rotation, (h,w)) #Rotates lena
        
        #TRANSLATION
        translation = np.array([[1, 0, random_t],[0 , 1, random_t]])
        img = cv2.warpAffine(img, translation, (h, w))
        return img

    #SAMPLE AUGMENT EXAMPLE ON LENA/IMG in DATASET ——————————————————————————————————————————————————————————————————————————————————————

    #Get image and initial parameters
    #lena = plt.imread('code/lena.png')
    #lena = augment(lena)

    #Display image
    #plt.title("Sample Augmentation — Lena")
    #plt.imshow(lena)
    #plt.show()
    mat = loadmat('code/mnist.mat')
    x = mat['trainX']
    randImg = np.reshape(x[random.randint(0,len(x))],(28,28))
    randImg = augment(randImg)

    #Display image
    plt.title("Sample Augmentation — Dataset")
    plt.imshow(randImg)
    plt.show()

    #AUGMENT ON DATASET N = 100 ———————————————————————————————————————————————————————————————————————————————————————————————————
    N = 100
    x = mat['trainX']
    testX = mat['testX']
    y = mat['trainY']
    testY = mat['testY']

    dataX = []
    dataY = []

    while (N>0):
        randInt = random.randint(0,len(x)-1)
        if y[randInt] == 6 or y[randInt] == 8:
            randImg = x[randInt]
            randImg = np.reshape(randImg,(28,28))
            randImg = augment(randImg)
            randImg = np.reshape(randImg,(784))
            dataX.append(randImg)
            if y[randInt] == 6:
                dataY.append(0)
            else:
                dataY.append(1)
        N-=1

    #RUN DATA WITH REGRESSION MODEL

    DELTA = 0.9 #Step-size/Learning Rate
    SIGMA = 5 #Scalar
    EPSILON = 8 #CONVERGENCE INTERVAL based on gradient
    STEPS = 25
    mat=loadmat('code/mnist.mat')

    col = 1
    dataX = np.array(dataX)
    dataY = np.array([dataY])
    BETA = np.zeros((dataX.shape[1])) #Initial beta value
    l = 1
    gradient = np.zeros(BETA.shape)
    plotY = []
    swatch = True

    #PART F: RERUN REGRESSION USING 6/8 instead of 0/1 ——————————————————————————————————————————————————————————————————————
    for iters in range (STEPS):
        for c in range (0, 784): #Image pixels
            l = 0.3
            for r in range (0, len(dataX)): #Num of images
                try:
                    l += ((dataY[0][r]) - math.exp(-np.dot(np.array(dataX[r]).T,BETA)) / (1 + (math.exp(-np.dot(np.array(dataX[r]).T,BETA))))) * dataX[r][c]
                except OverflowError:
                    l += 0
            gradient[c] = l
            
        gradient += BETA/(SIGMA**2)
        plotY.append(magnitude(gradient))
        BETA -= DELTA * gradient
        swatch = False
    plotX = [i for i in range(len(plotY))]

    #Getting Testing Data
    tX = []
    tY = []
    for i in range(len(testX)):
        arr = testX[i].tolist()
        if testY[i][0] == 6:
            tX.append(arr)
            tY.append(0)
            col+=1
        if testY[i][0] == 8:
            tX.append(arr)
            tY.append(1)

    tX = np.array(tX)
    tY = np.array([tY])
    
    #Generate predictions based on BETA and check if our predictions match our expected value
    predictions = np.array([])
    error = 0
    for i in range(len(tX)):
        try:
            predictions = np.append(predictions, np.round(1/(1+math.exp(np.dot(tX[i], BETA.T)))))
            if predictions[i] != tY[0][i]:
                error += 1
        except OverflowError:
            predictions = np.append(predictions, 0)
            error += 1
            
    #Print error/step-size/sigma
    error /= len(tX) 
    error *=  100 #Error percent
    print("Error: "+str(error)+"%")
    
    #GRADIENT PLOT
    #plt.plot(plotX, plotY)
    #plt.show()

    print("Step-size:", DELTA)
    print("Sigma:", SIGMA)
       

    #AUGMENT ON DATASET N = 300 ———————————————————————————————————————————————————————————————————————————————————————————————————
    N = 300

    #AUGMENT ON DATASET N = 600 ———————————————————————————————————————————————————————————————————————————————————————————————————
    N=600
    