import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Solution: 
    #Determines Principal Component Analysis
    def pca(self,x):
        # Normalize the input matrix
        x = (x - x.mean()) / x.std()

        # Determine the covariance matrix
        x = np.matrix(x)
        cov = (np.transpose(x) * x) / x.shape[1]

        # Perform Singular-Value Decomposition
        U, S, V = np.linalg.svd(cov)

        return U, S

    def main(self):
        # Load data from mnist.mat
        data = loadmat('MINIST_Q2/mnist.mat')
        input = data['trainX']

        # Get the principal components
        eigenvecs, eigenvals = self.pca(input)

        #PART A ————————————————————————————————––––––—————————–––———––———–––––––——————————————————

        # Plot the eigenvalues
        eigen_logs = np.log(eigenvals)
        plt.plot(eigen_logs)
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title("Eigenvalues With Logarithm Fit", fontsize=10)
        plt.show()

        #PART B ————————————————————————————————––––––—————————–––———––———–––––––——————————————————
        total_variance = eigenvals.sum()
        current_variance = 0
        p_variance = []
        k = 0

        #Determine principal components that retain 90% of variance
        while (float(current_variance / total_variance) < 0.9):
            current_variance += eigenvals[k]
            p_variance.append(float(current_variance/total_variance))
            k += 1

        # Plot the principal components that retain 90% of the variance, then print number of principal components needed
        #eigen_logs = np.log(p_components)
        plt.plot(p_variance)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('% Variance')
        plt.title("Number of principal components needed to retain 90% of the variance", fontsize=9)
        plt.show()
    
        print("Number of principal components needed to retain 90% of the variance: ", len(p_variance))

        #PART C ————————————————————————————————––––––—————————–––———––———–––––––——————————————————
        i10eigenvecs = eigenvecs[:, :10]
        fig, ax = plt.subplots(2, 5, figsize=(10, 5))

        # Plot the first 10 eigenvectors
        for i in range(2):
            for j in range(5):
                ax[i][j].imshow(i10eigenvecs[:, i * 5 + j].reshape(28, 28), cmap="gray")
                title = "Eigenvector " + str(5*i + j + 1)
                ax[i][j].set_title(title)
                ax[i][j].axis('off')
        plt.show()

sol = Solution()
sol.main()

