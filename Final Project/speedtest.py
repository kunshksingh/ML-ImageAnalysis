import time 
start = time.time()
e = 0
for i in range (10):
    print(((time.time()-start)),"seconds")
    for j in range (100):
        for k in range (100):
            for l in range (100):
                for m in range (100):
                    e += 1
print("Iterations:",e)