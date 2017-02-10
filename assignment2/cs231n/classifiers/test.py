import numpy as np

c = np.ones((2,4,4))
c[1][3] = 7

print np.argmax(c)
