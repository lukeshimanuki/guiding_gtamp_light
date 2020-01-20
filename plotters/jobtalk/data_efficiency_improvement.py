from matplotlib import pyplot as plt


ind = [1, 2]
plt.bar(ind, [90, 315])
#plt.title('Number of planning episodes required to solve all problems')
plt.xticks(ind, ('Ours', 'state-of-the-art\n planner'))
plt.xticks(fontsize=20)
plt.ylabel('planning time (sec)', fontsize=20)
plt.show()

ind = [1, 2]
plt.bar(ind, [10, 35])
#plt.title('Number of planning episodes required to solve all problems')
plt.xticks(ind, ('Ours', 'Benchmark'))
plt.xticks(fontsize=20)
plt.ylabel('n planning episodes', fontsize=20)
plt.show()


