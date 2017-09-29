import pandas as pd
import numpy as np
from scipy.spatial import distance
import sys
import json
import matplotlib.pyplot as plt

df = pd.DataFrame.from_csv("city_results.csv")

df = df.sort_values('Airbnb 1 month', ascending = False)

x = np.arange(df.shape[0])

print df

plt.style.use('fivethirtyeight')
#plt.style.use('ggplot')

import brewer2mpl
bmap = brewer2mpl.get_map('Set2','qualitative',3 ,reverse=True)
colors = bmap.mpl_colors

csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}

ax = df.plot(x, y=["Airbnb 1 month", "Regular 1 month"], color=colors, alpha=0.8, kind="bar")
#ax.set_xticks(x + 0.4)
ax.set_xticklabels(list(df.index.values), rotation="vertical")
plt.tight_layout()

ax.set_ylabel('Amount in Euro', **hfont)
plt.suptitle("Should you rent out your flat to tourists or locals?", fontsize=36, fontweight='bold', **csfont)

plt.rcParams["font.family"] = "cursive"

plt.matplotlib.rcParams.update({'font.size': 34})

plt.show()

sys.exit()

fig, ax = plt.subplots()
ax.scatter(df["Salary 1 month"], df['Airbnb 1 month'])
ax.set_xlabel('Salary 1 month')
ax.set_ylabel('Airbnb 1 month')
plt.show()

# positions of each tick, relative to the indices of the x-values
#ax.set_xticks(np.interp(new_ticks, s.index, np.arange(s.size)))

# labels
#ax.set_xticklabels(new_ticks)
