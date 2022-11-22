Welcome to my third (and final) problem-set for 4501 MLIA! :)

Feel free to contact me at kunshksingh@gmail.com for any comments/concerns. (Quick tip: If requirements.txt doesn't work, please check which Python Interpreter you are using; this issue happened to me when I started on pset1)

PDF Submission
---
My main PDF submission is in the root and is known as "bca2nk_A3_report.pdf". ps3.pdf is just the questions.

Table of Contents
---
Problem 1: 2-7
Problem 2: 8-9
Problem 3: TBD

Image Submissions
---
N/A

Code Submissions
---
All code can be found in the "code" folder. "regression01.py" is my code for parts c-e for question 1, and "regression68.py" is my code for part f of question 1. The solutions to parts a-b for question 1 can be found in my report. "regPCA.py" is my code for question 2. "augmentation.py" is my attempt at code for question 3.

For both regression01.py and regression68.py, running will print the sigma/step-size values and the error for the regression model. Please be patient with both of these regression models! They can take up to a minute to run.

For regPCA.py, running will print the sigma/step-size values, the error for each regression model, and the time taken to train regression model for all 3 PC=10,20,30, in that order. 

For augmentation.py, running will initially showcase my translation/rotation of a sample image from my dataset. I tried to implement the regression for N=100, N=300, and N=600, but depending on what I submit, there may be some errors or I may have not gotten around to fully implement it. It is not included in my report.

If receiving "ValueError: the input array must have size 3 along `channel_axis`, got (785, )" for any of the regression python files, please ensure you have the proper requirements installed properly. This error pops up when the dependencies are not installed properly.

---
Thank you once again for looking at my third problem set! Implementing that initial algorithm for gradient ascent took a while to transcribe into gradient descent for simplicity sake/to get a much lower error for the problem, but I finally feel like I'm learning something! Thank you so much for a wonderful 3 psets, and I'm looking forward to the final project :).
