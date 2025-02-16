import pandas as pd
from matplotlib import pyplot as plt


#PLOT 1__________________________________________________________________________________________________________________________

df1 = pd.read_csv("Variance_mean.txt")

plt.subplot(1,2,1)
plt.gca().set_facecolor('papayawhip')
plt.plot(df1.iloc[:,0], df1.iloc[:,1], color = 'blue')
plt.title("Mean of the empirical distribution of diff across the variance of the simulated data.", fontsize = 7)
plt.xlabel("Variance of simulated data")
plt.ylabel("Empirical mean")
plt.grid()

plt.subplot(1,2,2)
plt.gca().set_facecolor('papayawhip')
plt.plot(df1.iloc[:,0], df1.iloc[:,2], color = 'red')
plt.title("Variance of the empirical distribution of diff across the variance of the simulated data.", fontsize = 7)
plt.xlabel("Variance of simulated data")
plt.ylabel("Empirical variance")
plt.grid()


text1 = '\n'.join(["1.) For each sample size, the bootstrap",
                   "    model was run 1000 times",
                  " resulting in the empirical distribution",
                   "    of the diff statistic",
                  " diff = MSE_orig - MSE_boot, where",
                  " MSE_orig: MSE of original model",
                  " MSE_boot: MSE of bootstrap model",
                  "",
                  "2.) The sample size of the train and test data: n = 500"])

text2 = '\n'.join(["Parameters for running the",
                   "bootstrap model:",
                   "n_iter = 1000",
                   "location = 'mean'"])

plt.gcf().text(0.15, 0.40, text1, transform=plt.gcf().transFigure, fontsize=8,
              verticalalignment='top', bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=1'))

plt.gcf().text(0.65, 0.65, text2, transform=plt.gcf().transFigure, fontsize=8,
              verticalalignment='top', bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=1'))


plt.show()


# PLOT2_______________________________________________________________________________________
df2 = pd.read_csv("Variance_median.txt")

plt.subplot(1,2,1)
plt.gca().set_facecolor('papayawhip')
plt.plot(df2.iloc[:,0], df2.iloc[:,1], color = 'blue')
plt.title("Mean of the empirical distribution of diff across the variance of the simulated data.", fontsize = 7)
plt.xlabel("Variance of simulated data")
plt.ylabel("Empirical mean")
plt.grid()

plt.subplot(1,2,2)
plt.gca().set_facecolor('papayawhip')
plt.plot(df2.iloc[:,0], df2.iloc[:,2], color = 'red')
plt.title("Variance of the empirical distribution of diff across the variance of the simulated data.", fontsize = 7)
plt.xlabel("Variance of simulated data")
plt.ylabel("Empirical variance")
plt.grid()


text1 = '\n'.join(["1.) For each sample size, the bootstrap",
                   "    model was run 1000 times",
                  " resulting in the empirical distribution",
                   "    of the diff statistic",
                  " diff = MSE_orig - MSE_boot, where",
                  " MSE_orig: MSE of original model",
                  " MSE_boot: MSE of bootstrap model",
                  "",
                  "2.) The sample size of the train and test data: n = 500"])

text2 = '\n'.join(["Parameters for running the",
                   "bootstrap model:",
                   "n_iter = 1000",
                   "location = 'median'"])

plt.gcf().text(0.15, 0.40, text1, transform=plt.gcf().transFigure, fontsize=8,
              verticalalignment='top', bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=1'))

plt.gcf().text(0.65, 0.65, text2, transform=plt.gcf().transFigure, fontsize=8,
              verticalalignment='top', bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=1'))


plt.show()




# PLOT3_____________________________________________________________________________________________________________________________

plt.subplot(1,2,1)
plt.gca().set_facecolor('papayawhip')
plt.plot(df1.iloc[:,0], df1.iloc[:,1], color = 'blue', label = 'mean')
plt.plot(df2.iloc[:,0], df2.iloc[:,1], color = 'red', label = 'median')
plt.title("Mean of the empirical distribution of diff across the variance of the simulated data.", fontsize = 7)
plt.xlabel("Variance of simulated data")
plt.ylabel("Empirical mean")
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.gca().set_facecolor('papayawhip')
plt.plot(df1.iloc[:,0], df1.iloc[:,2], color = 'blue', label = 'mean')
plt.plot(df2.iloc[:,0], df2.iloc[:,2], color = 'red', label = 'median')
plt.title("Variance of the empirical distribution of diff across the variance of the simulated data.", fontsize = 7)
plt.xlabel("Variance of simulated data")
plt.ylabel("Empirical variance")
plt.grid()
plt.legend()

plt.show()