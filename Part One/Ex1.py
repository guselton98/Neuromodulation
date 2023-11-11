# NEUROMODULATION WORKSHOP
# PART 1 - EXERCISE 1: ARITHMETIC AND OUTPUT (PRINTING TO CONSOLE, WRITING TO FILE)

#----------------------------------------------------------------------------------------------------------------------
# IMPORTING LIBRARIES

# Any libraries referenced within our code file, must be imported at the top.
# Import the numpy library, so we can use those functions in our code.
import numpy

# If you want to shorten the reference name or get specific about which functions
# to use within a library, you could also type:
# import numpy as np
# OR:
# from numpy import array

#----------------------------------------------------------------------------------------------------------------------
# MULTIPLYING SCALARS

A = 1
B = 2.01
C = A*B
print("C is equal to", C)

#----------------------------------------------------------------------------------------------------------------------
# MULTIPLYING ARRAYS USING VECTORISATION

D = numpy.array([1, 5])   # 'numpy' is used to denote the array() function used is specifically from the numpy library.
E = 2.01
F = D*E
print("F is equal to ", F)

#----------------------------------------------------------------------------------------------------------------------
# WRITE TO A FILE
# The following is what is known as 'Object Oriented Programming':

G = open("temp.txt", "w")  # Creates an object named 'G' that stores our file name and instruction to write to it
H = "C is equal to " + str(C)
I = "F is equal to " + str(F)
G.write(H + "\n" + I)   # We're applying the functions write() and close() to our object 'G' for processing.
G.close()

#----------------------------------------------------------------------------------------------------------------------