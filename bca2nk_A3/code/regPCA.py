import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cv2
from scipy.io import loadmat
import time
class Reg01:
   
   #FUNCTIONS FROM PS1 —————————————————————————————————————————————————
    def magnitude(self,matrix): #Determines the magnitude of a 2D matrix
            return np.linalg.norm(matrix,ord=2)

    #PCA FROM PS2 —————————————————————————————————————————————————
    def pca(self,x):
        #Determines Principal Component Analysis
        # Normalize the input matrix
        x = (x - x.mean()) / x.std()

        # Determine the covariance matrix
        x = np.matrix(x)
        cov = (np.transpose(x) * x) / x.shape[1]

        # Perform Singular-Value Decomposition
        U, S, V = np.linalg.svd(cov)

        return U, S

    def main(self,mat=loadmat('code/mnist.mat')):
    #PCA FROM PS2 —————————————————————————————————————————————————
        # Load data from mnist.mat
        data = mat
        input = data['trainX']

        # Get the principal components
        eigenvecs, eigenvals = self.pca(input)


    #RUN REGRESSION WITH PC = 10 ———————————————————————————————————————————————————————————————————————————————
        DELTA = 0.8 #Step-size/Learning Rate
        SIGMA = 10 #Scalar
        EPSILON = 8 #CONVERGENCE INTERVAL based on gradient
        #STEPS = 10
    
        x = mat['trainX']
        testX = mat['testX']
        y = mat['trainY']
        testY = mat['testY']

        dataX = []
        dataY = []

        col = 1

        for i in range(len(x)):
            arr = x[i].tolist()
            if y[i][0] == 0:
                dataX.append(arr)
                dataY.append(0)
                col+=1
            if y[i][0] == 1:
                dataX.append(arr)
                dataY.append(1)

        dataX = np.array(dataX)
        dataY = np.array([dataY])
        
        l = 1
        
        plotY = []
        swatch = True #makes while loop a do while loop

        #DOWNSCALE IMAGES TO PC = 10
        
        dataDownsized = []
        for img in dataX:
            nft = []
            for e in range(10):
                nft.append(np.matmul(eigenvecs[e],img))
            dataDownsized.append(nft)
    
        dataX = dataDownsized
        dataX = np.squeeze(np.squeeze(np.array(dataX)))

        BETA = np.zeros((dataX.shape[1])) #Initial beta value
        gradient = np.zeros(BETA.shape)

        #Main regression method
        start = time.time()
        while(swatch or self.magnitude(gradient)>EPSILON):
            for c in range (0, 10): #Image pixels
                l = 0.3
                for r in range (0, len(dataX)): #Num of images
                    try:
                        l += ((dataY[0][r]) - math.exp(-np.dot(np.array(dataX[r]).T,BETA)) / (1 + (math.exp(-np.dot(np.array(dataX[r]).T,BETA))))) * dataX[r][c]
                    except OverflowError:
                        l += 0
                gradient[c] = l
                
            gradient += BETA/(SIGMA**2)
            plotY.append(self.magnitude(gradient))
            BETA -= DELTA * gradient
            swatch = False
        end = time.time()
        plotX = [i for i in range(len(plotY))]

        #Getting Testing Data
        tX = []
        tY = []
        for i in range(len(testX)):
            arr = testX[i].tolist()
            if testY[i][0] == 0:
                tX.append(arr)
                tY.append(0)
                col+=1
            if testY[i][0] == 1:
                tX.append(arr)
                tY.append(1)

        tX = np.array(tX)
        tY = np.array([tY])
        dataDownsized = []
        for img in tX:
            nft = []
            for e in range(10):
                nft.append(np.matmul(eigenvecs[e],img))
            dataDownsized.append(nft)

        tX = dataDownsized
        tX = np.squeeze(np.squeeze(np.array(tX)))

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
        
        print("Error for PC=10: "+str(error)+"%")

        #ERROR PLOT
        #plt.plot(plotX, plotY)
        #plt.show()

        print("Step-size:", DELTA)
        print("Sigma:", SIGMA)
        print("Training time (in seconds)",(end-start),'\n')

    #RUN REGRESSION WITH PC = 20 ———————————————————————————————————————————————————————————————————————————————
        DELTA = 0.8 #Step-size/Learning Rate
        SIGMA = 10 #Scalar
        EPSILON = 8 #CONVERGENCE INTERVAL based on gradient
        #STEPS = 10
    
        x = mat['trainX']
        testX = mat['testX']
        y = mat['trainY']
        testY = mat['testY']

        dataX = []
        dataY = []

        col = 1

        for i in range(len(x)):
            arr = x[i].tolist()
            if y[i][0] == 0:
                dataX.append(arr)
                dataY.append(0)
                col+=1
            if y[i][0] == 1:
                dataX.append(arr)
                dataY.append(1)

        dataX = np.array(dataX)
        dataY = np.array([dataY])

        l = 1
        plotY = []
        swatch = True #makes while loop a do while loop

        #DOWNSCALE IMAGES TO PC = 20
        dataDownsized = []
        for img in dataX:
            nft = []
            for e in range(20):
                nft.append(np.matmul(eigenvecs[e],img))
            dataDownsized.append(nft)
    
        dataX = dataDownsized
        dataX = np.squeeze(np.squeeze(np.array(dataX)))

        BETA = np.zeros((dataX.shape[1])) #Initial beta value
        gradient = np.zeros(BETA.shape)
        #Main regression method
        start=time.time()
        while(swatch or self.magnitude(gradient)>EPSILON):
            for c in range (0, 20): #Image pixels
                l = 0.3
                for r in range (0, len(dataX)): #Num of images
                    try:
                        l += ((dataY[0][r]) - math.exp(-np.dot(np.array(dataX[r]).T,BETA)) / (1 + (math.exp(-np.dot(np.array(dataX[r]).T,BETA))))) * dataX[r][c]
                    except OverflowError:
                        l += 0
                gradient[c] = l
                
            gradient += BETA/(SIGMA**2)
            plotY.append(self.magnitude(gradient))
            BETA -= DELTA * gradient
            swatch = False
        end = time.time()
        plotX = [i for i in range(len(plotY))]

        #Getting Testing Data
        tX = []
        tY = []
        for i in range(len(testX)):
            arr = testX[i].tolist()
            if testY[i][0] == 0:
                tX.append(arr)
                tY.append(0)
                col+=1
            if testY[i][0] == 1:
                tX.append(arr)
                tY.append(1)

        tX = np.array(tX)
        tY = np.array([tY])

        dataDownsized = []
        for img in tX:
            nft = []
            for e in range(20):
                nft.append(np.matmul(eigenvecs[e],img))
            dataDownsized.append(nft)
    
        tX = dataDownsized
        tX = np.squeeze(np.squeeze(np.array(tX)))
    
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
        
        print("Error for PC=20: "+str(error)+"%")

        #ERROR PLOT
        #plt.plot(plotX, plotY)
        #plt.show()

        print("Step-size:", DELTA)
        print("Sigma:", SIGMA)
        print("Training time (in seconds)",(end-start),'\n')
        
        #RUN REGRESSION WITH PC = 30 ———————————————————————————————————————————————————————————————————————————————
        DELTA = 0.8 #Step-size/Learning Rate
        SIGMA = 10 #Scalar
        EPSILON = 8 #CONVERGENCE INTERVAL based on gradient
        #STEPS = 10
    
        x = mat['trainX']
        testX = mat['testX']
        y = mat['trainY']
        testY = mat['testY']

        dataX = []
        dataY = []

        col = 1

        for i in range(len(x)):
            arr = x[i].tolist()
            if y[i][0] == 0:
                dataX.append(arr)
                dataY.append(0)
                col+=1
            if y[i][0] == 1:
                dataX.append(arr)
                dataY.append(1)

        dataX = np.array(dataX)
        dataY = np.array([dataY])
        l = 1
        plotY = []
        swatch = True #makes while loop a do while loop

        #DOWNSCALE IMAGES TO PC = 30
        dataDownsized = []
        for img in dataX:
            nft = []
            for e in range(30):
                nft.append(np.matmul(eigenvecs[e],img))
            dataDownsized.append(nft)
    
        dataX = dataDownsized
        dataX = np.squeeze(np.squeeze(np.array(dataX)))

        BETA = np.zeros((dataX.shape[1])) #Initial beta value
        gradient = np.zeros(BETA.shape)

        #Main regression method
        start = time.time()
        while(swatch or self.magnitude(gradient)>EPSILON):
            for c in range (0, 30): #Image pixels
                l = 0.3
                for r in range (0, len(dataX)): #Num of images
                    try:
                        l += ((dataY[0][r]) - math.exp(-np.dot(np.array(dataX[r]).T,BETA)) / (1 + (math.exp(-np.dot(np.array(dataX[r]).T,BETA))))) * dataX[r][c]
                    except OverflowError:
                        l += 0
                gradient[c] = l
                
            gradient += BETA/(SIGMA**2)
            plotY.append(self.magnitude(gradient))
            BETA -= DELTA * gradient
            swatch = False
        end = time.time()
        plotX = [i for i in range(len(plotY))]

        #Getting Testing Data
        tX = []
        tY = []
        for i in range(len(testX)):
            arr = testX[i].tolist()
            if testY[i][0] == 0:
                tX.append(arr)
                tY.append(0)
                col+=1
            if testY[i][0] == 1:
                tX.append(arr)
                tY.append(1)

        tX = np.array(tX)
        tY = np.array([tY])
        dataDownsized = []
        for img in tX:
            nft = []
            for e in range(30):
                nft.append(np.matmul(eigenvecs[e],img))
            dataDownsized.append(nft)
    
        tX = dataDownsized
        tX = np.squeeze(np.squeeze(np.array(tX)))
    
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
        
        print("Error for PC=30: "+str(error)+"%")

        #ERROR PLOT
        #plt.plot(plotX, plotY)
        #plt.show()

        print("Step-size:", DELTA)
        print("Sigma:", SIGMA)
        print("Training time (in seconds)",(end-start))

        

reg = Reg01()
reg.main()