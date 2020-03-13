from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(2)
money = ['Guided planner', 'State-of-the-art planner']


def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)


formatter = FuncFormatter(millions)

fig, ax = plt.subplots()
#ax.yaxis.set_major_formatter(formatter)
plt.bar(x, [164, 318])
plt.xticks(x, ('Guided \n planner', 'State-of-the-art \n planner'))
plt.ylabel('Planning times in seconds',fontsize=14)
import pdb;pdb.set_trace()

