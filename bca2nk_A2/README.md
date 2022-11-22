Welcome to my second problem-set for 4501 MLIA! :)

Feel free to contact me at kunshksingh@gmail.com for any comments/concerns. (Quick tip: If requirements.txt doesn't work, please check which Python Interpreter you are using; this issue happened to me when I started on pset1)

PDF Submission
---
My main PDF submission is in the root and is known as "bca2nk_A2_report.pdf". ps2.pdf is just the questions.

If you would just like to see my Problem 3/4 for my geodesic shooting, then *skip to page 10*.

However, if you are interested in my thought process for Problem 1 and Problem 2, see pages 1-9 in the PDF.

Table of Contents
---
Problem 1: 2-6

Problem 2: 6-10

Problem 3: 10-15

Problem 4: 15-17

Image Submissions
---
My 1.jpeg titled "1a Closed-Form Eulers" is my handwritten solution for the closed form differential equation for part 1a. My 5 .png titled "1 0.1step", "1 0.05 step", "1 0.01 step","1 0.005 step", and "1 0.001 step" are graphs for Problem 1 parts b&c. This can also be seen in pages 1, 5-6 in the PDF, respectively.

My 3 .png titled "2 allEigen", "2 pcaComps", and "2 eigenVecs" are my respective outputs to parts 2A, 2B, and 2C in Problem 2. A further explanation for these solutions can be see from pages 6-10 in the PDF.

My 1 .png titled "3 results" displays image inputs and their respective outputs upon undergoing geodesic shooting. A further explanation can be seen in pages 10-15 of the PDF.

My 1 .png titled "4 results" displays image inputs and their respective outputs upon undergoing geodesic shooting. A further explanation can be seen in pages 15-17 of the PDF.

Code Submissions
---
All code can be found in the "code" folder. "Eulers.py" is my code for Problem 1 and has annotations denoting my thought process and sections denoting each part (b, and c). "PCA.py" includes my principal component analysis for Problem 2 and also has annotations describing my thought processes and sections denoting each part (a, b, and c). "GeodesicShooting.py" is my code for Problem 3 sectioned out to with many functions, constants, and more. "GeodesicShooting2.py" is my code for Problem 4 and is sectioned out similarly to the previous GeodesicShooting problem.

For Eulers.py, running will show the output for each step size in h = {0.1,0.05,0.01,0.005,0.001}. Closing out each figure will show the next graph with the next respective step size in the set of h. The graphs plot the Euler's method (in blue) against the close form solution determined in part a (in orange).

For PCA.py, running it once will give the proper output for 2A. Closing out the output (using the x in the top left) for 2A will show the proper output for 2B. Closing it out one more time will show the proper output for 2C.

If receiving "ValueError: the input array must have size 3 along `channel_axis`, got (319, 320)" for  GeodesicShooting.py/GeodesicShooting2.py, please ensure you have the proper requirements installed properly. This error pops up when the dependencies are not installed properly.

---
Thank you once again for looking at my second problem set! Understanding/implementing/writing about geodesic shooting took me even longer than gradient descent; geodesic shooting alone took me ~30 hours so I do not know how much my entire problem set took. Though I enjoyed problem set 1 more, working with image registration and diffeomorphic images was its own thrill :). Having someone read my hard work always means a lot to me!
