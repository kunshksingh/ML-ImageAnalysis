import math
import matplotlib.pyplot as plt 
import numpy as np

class Solution:

    def func(self,x,y): #func is the derative of the function based on the given differential equation
        return 2 - math.exp((-4 * x)) - 2 * y

    def closed(self,x): #close form solution, giving us our solution function 
        return 1 + (1/2) * math.exp((-4 * x)) - (1/2) * math.exp((-2 * x))

    def main(self):
        y0 = 1

        #PART B ——————————————————————————————————————————————————————————
        t = [1,2,3,4,5] #Given set
        max_step = t[-1] #Last in t
    
        #PART C ——————————————————————————————————————————————————————————
        step_size = [0.1,0.05,0.01,0.005,0.001] #Delta t = given h
        for change in step_size:
            self.eulers(y0, max_step, change)


    #Find Euler's
    def eulers(self, y0, maxStep, step):#x0 is first step value
                                        #y0 is initial y value
                                        #Step is the step-size increment
                                        #maxStep is the largest step we take from our initial value
        yVals = np.array([])
        xVals = np.array([])
        t = [1.0,2.0,3.0,4.0,5.0]
        y_t = []
        y_approx = y0
        inv = 1/step
        
        #Main Euler's function
        for i in range(0,int((maxStep/step)+1),int(step*inv)): #[0,maxStep] inclusive
            if ((i/inv) in t):
                y_t.append(y_approx)
            yVals = np.append(yVals,y_approx)
            y_approx += self.func((i/inv),y_approx)*(1/inv) #Perform main Euler's formula (y(k+1) = y(k) + dy/dx * delta(x)) 
            xVals = np.append(xVals,(i/inv)) 
           

        #Get Closed Form Values
        yVals2 = np.array([])
        for x in xVals:
            yVals2 = np.append(yVals2,self.closed(x))

        #Show Results
        plt.plot(xVals, yVals, label="Euler's")
        plt.plot(xVals, yVals2, label="Closed Form")
        plt.plot(t, y_t, 'b*')
        plt.xlabel("Time (t)")
        plt.ylabel("F(t)")
        title = "Eulers graph with " + str(step) + " step size vs. the closed form solution"
        plt.title(title, fontsize=10)
        plt.legend(loc="upper left")
        plt.show()
        

#Calling solution class
sol = Solution()
sol.main()