import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style("white",{'font.family':'serif','font.serif':'Palatino'})
sns.set_palette("colorblind")
ax = sns.lineplot(
    0,
    1,
    data=np.array([[1, 2], [3, 4]]),
)
ax.ylabel("Value")
# print(plt.rcParams)
plt.show()