import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.01, 5.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.exp(-t)
s3 = np.sin(4*np.pi*t)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, s1)
ax2.plot(t, s3)
f.savefig('test.pdf', format='pdf', bbox_inches='tight')
