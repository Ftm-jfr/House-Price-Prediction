import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder

# load data
df = pd.read_csv('house-prices.csv', sep=r'\s+')

# preprocessing
print(df.describe())
print("\n")
print(df.info())
print("\n")

# replacing non values with mean or median
print(f" before filling non values : {df.isna().sum()}")
df[[ "SqFt"]] = df[[ "SqFt"]].apply(lambda x: x.fillna(x.mean()), axis=0)
df[["Bedrooms", "Bathrooms","Offers"]] = df[["Bedrooms", "Bathrooms","Offers"]].apply(lambda x: x.fillna(x.median()), axis=0)
print(f" After  filling non values : {df.isna().sum()}")


# seperate train and test data
X = df[['SqFt', 'Bedrooms', 'Bathrooms', 'Offers', 'Brick', 'Neighborhood']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


def detect_outliers_iqr(df, numeric_columns):
    outliers = {}

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    return outliers

def detect_outliers_zscore(df, numeric_columns):
    df_zscore = df.copy()
    outliers= {}

    for col in numeric_columns:
        mean = df[col].mean()
        std = df[col].std()
        df_zscore[col] = (df[col] - mean) / std

        outliers[col] = df[(df_zscore[col] > 3) | (df_zscore[col] < -3)]

    return  outliers


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
outliers_iqr = detect_outliers_iqr(df, numeric_columns)

for col, outlier_data in outliers_iqr.items():
    if not outlier_data.empty:
        print(f"Outliers in {col} based on IQR:")
        print(outlier_data)
    else:
        print(f"No outliers in {col} based on IQR")

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
outliers_zscore = detect_outliers_zscore(df, numeric_columns)
for col, outlier_data in outliers_zscore.items():
    if not outlier_data.empty:
        print(f"Outliers in {col} based on Z-Score:")
        print(outlier_data)
    else:
        print(f"No outliers in {col} based on Z-Score")


# showing relation between data
y_price = df["Price"]
sub_numeric_columns = [col for col in df.select_dtypes(include=["number"]).columns if col != "Price"]
# scatter
plot_num = len(sub_numeric_columns)
plt.figure(figsize=(5 * plot_num, 5))
for i, col in enumerate(sub_numeric_columns):
    plt.subplot(1, plot_num, i + 1)
    sns.scatterplot(x=df[col], y=y_price)
plt.tight_layout()
plt.show()
# heatmap
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

plt.title("Heatmap of Feature Correlations")
plt.show()

# regplot
plt.figure(figsize=(5 * plot_num, 5))
for i, col in enumerate(sub_numeric_columns):
    plt.subplot(1, plot_num, i + 1)
    sns.regplot(x=df[col], y=y_price, scatter_kws={"s": 10}, line_kws={"color": "red"})

plt.tight_layout()
plt.show()

class LinearRegression:
    def __init__(self, alpha=0.01, num_iters=1000):
        self.alpha = alpha
        self.num_iters = num_iters
        self.w = None
        self.b = None

    def normalize(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    def compute_rmse(self, y_pred, y_true):
        mse = np.mean((y_pred - y_true) ** 2)
        return np.sqrt(mse)

    def compute_gradient(self, X, y):
        m = X.shape[0]
        y_pred = np.dot(X, self.w) + self.b
        error = y_pred - y
        dj_dw = np.dot(X.T, error) / m
        dj_db = np.sum(error) / m
        return dj_dw, dj_db

    def fit(self, X, y):
        X = self.normalize(X)
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        self.cost_history = []

        for i in range(self.num_iters):
            dj_dw, dj_db = self.compute_gradient(X, y)
            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db
            y_pred = np.dot(X, self.w) + self.b
            cost = self.compute_rmse(y_pred, y)
            self.cost_history.append(cost)

    def predict(self, X):
        X = (X - self.mean) / self.std
        return np.dot(X, self.w) + self.b


nonNumeric_columns = X_train.select_dtypes(include=['object', 'string']).columns

le = LabelEncoder()
for col in nonNumeric_columns:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

model = LinearRegression(alpha=0.01, num_iters=500)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse_test = model.compute_rmse(y_pred, y_test)

print("RSME on  test", rmse_test)
