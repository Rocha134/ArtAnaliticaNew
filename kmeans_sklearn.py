import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("ulabox_orders_with_categories_partials_2017.csv")

dfp = df[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]]

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

kmeans = KMeans(n_clusters=k).fit(df[["Food%", "Fresh%", "Drinks%", "Home%", "Beauty%", "Health%", "Baby%", "Pets%"]])
sns.scatterplot(data=df, x="weekday", y="hour", hue=kmeans.labels_)
plt.show()

cluster0=df[kmeans.labels_==0]
print(cluster0.describe())

cluster1=df[kmeans.labels_==1]
print(cluster1.describe())

cluster2=df[kmeans.labels_==2]
print(cluster2.describe())

cluster3=df[kmeans.labels_==3]
print(cluster3.describe())

cluster4=df[kmeans.labels_==4]
print(cluster4.describe())
# %%
sns.boxplot(data=cluster0[['Food%','Fresh%','Drinks%','Home%','Beauty%','Health%','Baby%','Pets%']])
sns.boxplot(data=cluster1[['Food%','Fresh%','Drinks%','Home%','Beauty%','Health%','Baby%','Pets%']])
sns.boxplot(data=cluster2[['Food%','Fresh%','Drinks%','Home%','Beauty%','Health%','Baby%','Pets%']])
sns.boxplot(data=cluster3[['Food%','Fresh%','Drinks%','Home%','Beauty%','Health%','Baby%','Pets%']])
sns.boxplot(data=cluster4[['Food%','Fresh%','Drinks%','Home%','Beauty%','Health%','Baby%','Pets%']])
# %%
sns.histplot(data=cluster0, x= 'weekday')
sns.histplot(data=cluster1, x= 'weekday')
sns.histplot(data=cluster2, x= 'weekday')
sns.histplot(data=cluster3, x= 'weekday')
sns.histplot(data=cluster4, x= 'weekday')
# %%
# from sklearn.tree import DecisionTreeClassifier, export_text
#
# tree = DecisionTreeClassifier()
# tree.fit(df[["weekday", "hour", "discount%", "total_items"]], kmeans.labels_)
# print(export_text(tree, feature_names=["weekday", "hour", "discount%", "total_items"]))
