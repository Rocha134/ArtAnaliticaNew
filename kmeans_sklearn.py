import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("Mall_Customers.csv")

dfp = df[["Annual_Income_(k$)", "Spending_Score"]]

ssd = []
ks = range(1,11)
for k in range(1,11):
    km = KMeans(n_clusters=k)
    km = km.fit(dfp)
    ssd.append(km.inertia_)

kneedle = KneeLocator(ks, ssd, S=1.0, curve="convex", direction="decreasing")
kneedle.plot_knee()
plt.show()

k = round(kneedle.knee)

print(f"Number of clusters suggested by knee method: {k}")

kmeans = KMeans(n_clusters=k).fit(df[["Annual_Income_(k$)", "Spending_Score"]])
sns.scatterplot(data=df, x="Annual_Income_(k$)", y="Spending_Score", hue=kmeans.labels_)
plt.show()


from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier()
tree.fit(df[["Age", "Annual_Income_(k$)", "Spending_Score"]], kmeans.labels_)
print(export_text(tree, feature_names=["Age", "Annual_Income_(k$)", "Spending_Score"]))