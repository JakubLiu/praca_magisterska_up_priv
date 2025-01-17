import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt


# PLOT 1_______________________________________________________________________________________________________________________________
"""
df = pd.read_csv("C:/Users/Qba Liu/Documents/STUDIA/BIOINF_MASTER_UPWR/PRACA_MAGISTERSKA/SYSTEMATIC_SIM/SAMPLE_SIZE/Sample_Size_sim_res.txt")


plt.subplot(1,2,1)
plt.gca().set_facecolor('papayawhip')
plt.plot(df.iloc[5:,0], df.iloc[5:,1], color = 'blue')
plt.title("Mean of the empirical distribution of diff across sample sizes.")
plt.xlabel("Sample size")
plt.ylabel("Empirical mean")
plt.grid()

plt.subplot(1,2,2)
plt.gca().set_facecolor('papayawhip')
plt.plot(df.iloc[5:,0], df.iloc[5:,2], color = 'red')
plt.title("Variance of the empirical distribution of diff across sample sizes.")
plt.xlabel("Sample size")
plt.ylabel("Empirical variance")
plt.grid()


text1 = '\n'.join(["For each sample size, the bootstrap",
                   "model was run 1000 times",
                  "resulting in the empirical distribution",
                   "of the diff statistic",
                  "diff = MSE_orig - MSE_boot, where",
                  "MSE_orig: MSE of original model",
                  "MSE_boot: MSE of bootstrap model"])

text2 = '\n'.join(["Parameters for running the",
                   "bootstrap model:",
                   "n_iter = 1000",
                   "location = 'mean'"])

plt.gcf().text(0.25, 0.65, text1, transform=plt.gcf().transFigure, fontsize=8,
              verticalalignment='top', bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=1'))

plt.gcf().text(0.65, 0.65, text2, transform=plt.gcf().transFigure, fontsize=8,
              verticalalignment='top', bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=1'))

#plt.show()
"""
#____________________________________________________________________________________________________________________________________

# PLOT 2 ______________________________________________________________________________________________________________________
'''
df = pd.read_csv("C:/Users/Qba Liu/Documents/STUDIA/BIOINF_MASTER_UPWR/PRACA_MAGISTERSKA/SYSTEMATIC_SIM/SAMPLE_SIZE/Sample_Size_sim_res_median.txt")
print(df.head())

plt.subplot(1,2,1)
plt.gca().set_facecolor('papayawhip')
plt.plot(df.iloc[5:,0], df.iloc[5:,1], color = 'blue')
plt.title("Mean of the empirical distribution of diff across sample sizes.")
plt.xlabel("Sample size")
plt.ylabel("Empirical mean")
plt.grid()

plt.subplot(1,2,2)
plt.gca().set_facecolor('papayawhip')
plt.plot(df.iloc[5:,0], df.iloc[5:,2], color = 'red')
plt.title("Variance of the empirical distribution of diff across sample sizes.")
plt.xlabel("Sample size")
plt.ylabel("Empirical variance")
plt.grid()

text1 = '\n'.join(["For each sample size, the bootstrap",
                   "model was run 1000 times",
                  "resulting in the empirical distribution",
                   "of the diff statistic",
                  "diff = MSE_orig - MSE_boot, where",
                  "MSE_orig: MSE of original model",
                  "MSE_boot: MSE of bootstrap model"])

text2 = '\n'.join(["Parameters for running the",
                   "bootstrap model:",
                   "n_iter = 1000",
                   "location = 'median'"])

plt.gcf().text(0.25, 0.65, text1, transform=plt.gcf().transFigure, fontsize=8,
              verticalalignment='top', bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=1'))

plt.gcf().text(0.65, 0.65, text2, transform=plt.gcf().transFigure, fontsize=8,
              verticalalignment='top', bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=1'))

plt.show()
'''

#__________________________________________________________________________________________________________________________________-

# PLOT 3_____________________________________________________________________________________________________________________________
df1 = pd.read_csv("C:/Users/Qba Liu/Documents/STUDIA/BIOINF_MASTER_UPWR/PRACA_MAGISTERSKA/SYSTEMATIC_SIM/SAMPLE_SIZE/Sample_Size_sim_res.txt")
df2 = pd.read_csv("C:/Users/Qba Liu/Documents/STUDIA/BIOINF_MASTER_UPWR/PRACA_MAGISTERSKA/SYSTEMATIC_SIM/SAMPLE_SIZE/Sample_Size_sim_res_median.txt")

plt.subplot(1,2,1)
plt.gca().set_facecolor('papayawhip')
plt.plot(df1.iloc[5:,0], df1.iloc[5:,1], color = 'blue', label = 'location: mean')
plt.plot(df2.iloc[5:,0], df2.iloc[5:,1], color = 'red', label = 'location: median')
plt.title("Mean of the empirical distribution of diff across sample sizes.")
plt.xlabel("Sample size")
plt.ylabel("Empirical mean")
plt.grid()

plt.subplot(1,2,2)
plt.gca().set_facecolor('papayawhip')
plt.plot(df1.iloc[5:,0], df1.iloc[5:,2], color = 'blue', label = 'location: mean')
plt.plot(df2.iloc[5:,0], df2.iloc[5:,2], color = 'red', label = 'location: median')
plt.title("Variance of the empirical distribution of diff across sample sizes.")
plt.xlabel("Sample size")
plt.ylabel("Empirical variance")
plt.grid()

plt.legend()

plt.show()