# Customer Segmentation using K-Means Clustering

## 📌 Project Overview

This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to segment customers based on their demographic and purchasing characteristics.

The analysis explores customer segments using:

* Age
* Annual Income
* Spending Score

The goal is to identify groups of customers with similar characteristics and understand how different features affect customer segmentation.

---

## 📊 Dataset

The dataset contains **200 customers** and the following features:

| Feature         | Description                           |
| --------------- | ------------------------------------- |
| `CustomerID`    | Unique customer identifier            |
| `Gender`        | Customer gender                       |
| `Age`           | Customer age                          |
| `AnnualIncome`  | Annual income in thousands of dollars |
| `SpendingScore` | Spending score from 1 to 100          |

### Dataset Summary

* **Rows:** 200
* **Features:** 5
* **Missing values:** None
* **Age:** 18–70
* **Annual Income:** 15k–137k
* **Spending Score:** 1–99

---

## 🛠️ Technologies & Libraries

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* K-Means Clustering

---

## 🔍 1. Data Exploration

The first step was to inspect the dataset using:

```python
df.head()
df.describe()
df.info()
df.isnull().sum()
```

The dataset contains no missing values, and the numerical features are ready for further analysis.

The column names were also simplified:

```python
df.rename(
    columns={
        'Annual Income (k$)': 'AnnualIncome',
        'Spending Score (1-100)': 'SpendingScore'
    },
    inplace=True
)
```

---

## 📈 2. Exploratory Data Analysis

Pair plots and scatter plots were used to investigate relationships between customer characteristics.

### Annual Income vs Spending Score

The relationship between annual income and spending score was visualized to identify potential customer groups.

```python
plt.scatter(
    df['AnnualIncome'],
    df['SpendingScore'],
    s=50
)
```

This visualization suggests that customers can be divided into several groups based on their income and spending behavior.

---

## 🤖 3. K-Means Clustering

### What is K-Means?

K-Means is an unsupervised machine learning algorithm that divides observations into a predefined number of clusters.

The algorithm works by:

1. Selecting initial cluster centroids.
2. Assigning each observation to the nearest centroid.
3. Updating the centroids.
4. Repeating the process until the clusters stabilize.

---

## 📐 4. Choosing the Number of Clusters

The **Elbow Method** was used to determine a suitable number of clusters.

The Within-Cluster Sum of Squares (WCSS), also known as inertia, was calculated for different values of `K`.

```python
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=0
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

The resulting WCSS values were plotted to identify the point where adding more clusters provides diminishing improvements.

---

# 📌 Experiment 1: Annual Income & Spending Score

The first clustering experiment used:

```python
X = df[['AnnualIncome', 'SpendingScore']]
```

The Elbow Method suggested using **5 clusters**.

```python
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=0
)

y_kmeans = kmeans.fit_predict(X)
df['Cluster'] = y_kmeans
```

### Visualization

The resulting clusters were visualized together with their centroids.

```python
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y_kmeans,
    s=50,
    cmap='viridis'
)
```

This segmentation highlights groups such as customers with:

* Low income and low spending
* Low income and high spending
* High income and low spending
* High income and high spending
* Medium income and medium spending

---

# 📌 Experiment 2: Age & Spending Score

A second experiment investigated whether customers could be segmented using **Age** and **Spending Score**.

```python
X = df[['Age', 'SpendingScore']]
```

The Elbow Method was applied again, resulting in **4 clusters**.

```python
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=0
)

y_kmeans = kmeans.fit_predict(X)

df['Cluster_Age'] = y_kmeans
```

The clusters were then visualized using Age and Spending Score.

This experiment provides a different perspective on customer behavior by focusing on the relationship between **age and spending habits** rather than income.

---

# 📌 Experiment 3: Age, Annual Income & Spending Score

Finally, all three numerical characteristics were combined:

```python
X = df[['Age', 'AnnualIncome', 'SpendingScore']]
```

The Elbow Method suggested **6 clusters** for this experiment.

```python
kmeans = KMeans(
    n_clusters=6,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=0
)

y_kmeans = kmeans.fit_predict(X)

df['Cluster_AgeIncomeSpend'] = y_kmeans
```

A 3D visualization was created to examine the resulting customer segments:

```python
ax.scatter(
    df['Age'],
    df['AnnualIncome'],
    df['SpendingScore'],
    c=df['Cluster_AgeIncomeSpend'],
    s=50,
    cmap='viridis'
)
```

### Features used

* **X-axis:** Age
* **Y-axis:** Annual Income
* **Z-axis:** Spending Score

This experiment provides a more comprehensive segmentation by considering demographic and behavioral characteristics simultaneously.

---

## 📊 Experiments Comparison

| Experiment   | Features                             | Number of Clusters |
| ------------ | ------------------------------------ | -----------------: |
| Experiment 1 | Annual Income + Spending Score       |                  5 |
| Experiment 2 | Age + Spending Score                 |                  4 |
| Experiment 3 | Age + Annual Income + Spending Score |                  6 |

The different experiments demonstrate how the choice of features affects the resulting customer segmentation.

---

## 💡 Key Insights

The analysis demonstrates that:

* Customers can be grouped according to similarities in their purchasing behavior.
* **Annual Income and Spending Score** provide a strong basis for identifying distinct customer segments.
* Adding **Age** creates a more multidimensional view of customer behavior.
* Different feature combinations lead to different optimal numbers of clusters.
* K-Means can be useful for discovering customer groups without predefined labels.

These segments could potentially support marketing strategies such as targeted promotions, customer profiling, and personalized campaigns.

---

## 🚀 Possible Improvements

Future versions of this project could include:

* Feature scaling using `StandardScaler`
* Comparison with other clustering algorithms such as DBSCAN and Hierarchical Clustering
* Evaluation using the **Silhouette Score**
* Analysis of cluster characteristics using group statistics
* Visualization with interactive Plotly charts
* Building a customer segmentation dashboard using **Power BI**
* Developing a simple recommendation or marketing strategy based on the identified clusters

---

## 📁 Project Structure

```text
Customer-Segmentation-KMeans/
│
├── Mall_Customers.csv
├── Customer_Segmentation.ipynb
├── README.md
└── images/
    ├── pairplot.png
    ├── income_spending.png
    ├── elbow_income_spending.png
    ├── customer_segments.png
    ├── age_spending.png
    ├── elbow_age_spending.png
    ├── age_segments.png
    ├── elbow_3_features.png
    └── 3d_customer_segments.png
```

---

## 🎯 Conclusion

This project demonstrates the use of **K-Means Clustering for customer segmentation** using Python.

By experimenting with different combinations of customer features, the project shows how unsupervised learning can uncover meaningful patterns in customer data without requiring predefined target labels.

The project also provides practical experience with:

* Data cleaning
* Exploratory Data Analysis
* Data visualization
* Unsupervised Machine Learning
* K-Means Clustering
* Elbow Method
* 2D and 3D visualization

---


