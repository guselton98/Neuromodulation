# Neuromodulation Workshop: EXE 3

# Import the numpy library
import numpy as np

# Dot Multiply Arrays
A = np.array([1, 5])
B = 2.01
C = A*B
print("C is equal to ", C)


# Write to a file
f = open("../temp.txt", "w")
s = "C is equal to " + str(C)
f.write(s)
f.close()