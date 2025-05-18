import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

db = pd.read_csv('Credit.csv')
db["Own"] = db["Own"].map({"Yes": 1, "No": 0})
db["Student"] = db["Student"].map({"Yes": 1, "No": 0})
db["Married"] = db["Married"].map({"Yes": 1, "No": 0})


for Region in  pd.unique(db['Region']):
	df=db[db['Region']==Region].drop(columns=["Region"])
	corr_matrix = df.corr()
	plt.figure(figsize=(10, 10))
	sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
	plt.title(f"Correlation Heatmap for Region: {Region}")
	plt.savefig(f"Credit_corr_matrix_{Region}.png")
	plt.show()

sns.boxplot(x = db['Cards'],y = db['Income'] )
plt.xlabel("Number of Credit Cards")
plt.ylabel("Income")
plt.title("Box Plot of Income for Each Number of Credit Cards")
plt.savefig(f"Income for Each Number of Credit Cards.png")
plt.show()


P_R = 0.60
P_P = 0.30
P_V = 0.10
P_B_R = 0.4
P_B_P = 0.7
P_B_V = 0.9

P_B = (P_B_R * P_R) + (P_B_P * P_P) + (P_B_V * P_V)
print(f"{round(P_B*100,2)}%")


P_D = 0.05
P_LS = 0.30
P_LS_D = 0.80
P_D_LS = (P_LS_D * P_D) / P_LS
print(f"{round(P_D_LS*100,2)}%")

n = 10
matrix = np.random.exponential(scale=1/2, size=(n, 1000))
Biased = np.sum((matrix - np.mean(matrix)) ** 2 ) / n
unBiased = np.sum((matrix - np.mean(matrix)) ** 2 ) / ( n - 1)
mean_Biased = np.mean(Biased)
mean_unBiased = np.mean(unBiased)
print(f"Mean of Based: {mean_Biased}")
print(f"Mean of UnBiased: {mean_unBiased}")

difference = mean_unBiased / mean_Biased
difference_Theoretical = n / (n - 1)
print(f"Difference: {round(difference,3)}")
print(f"Theoretical Difference: {round(difference_Theoretical,3)}")


n = 1000
lambda_val = 3
samples = np.random.exponential(scale=1/lambda_val, size=(n, 100))

true_mean = 1 / lambda_val
mean_samples = np.mean(np.mean(samples, axis=1), axis=0)
print(f"True Mean: {true_mean}")
print(f"Mean of Samples: {mean_samples}")

true_variance = 1 / (lambda_val ** 2)
var_samples = mean_samples ** 2
print(f"True Variance: {true_variance}")
print(f"Variance from Samples: {var_samples}")

true_sd = np.sqrt(true_variance)
print(f"True Standard Deviation: {true_sd}")
print(f"Standard Deviation from Samples: {mean_samples}")


