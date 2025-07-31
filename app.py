
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('ecoline_sales.csv')

# Data Preprocessing
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = pd.get_dummies(df, columns=['region'], drop_first=True)
df['month'] = pd.to_datetime(df['month'])
df['month_num'] = df['month'].dt.month

# Visualizations
st.title("Ecoline Sales Analysis and Prediction")

st.subheader("Sales Over Time")
fig1, ax1 = plt.subplots(figsize=(10,4))
sns.lineplot(x='month', y='sales', data=df, ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("Temperature vs Sales")
fig2, ax2 = plt.subplots()
sns.scatterplot(x='temperature', y='sales', data=df, ax=ax2)
st.pyplot(fig2)

st.subheader("Effect of Promotion on Sales")
fig3, ax3 = plt.subplots()
sns.boxplot(x='promotion', y='sales', data=df, ax=ax3)
st.pyplot(fig3)

st.subheader("Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

st.subheader("Average Sales by Region")
region_cols = [col for col in df.columns if 'region_' in col]
df_region = df[region_cols + ['sales']].groupby(df[region_cols].idxmax(axis=1)).mean()
fig5, ax5 = plt.subplots()
df_region.plot(kind='bar', legend=False, ax=ax5)
ax5.set_ylabel('Sales')
st.pyplot(fig5)

# Model Training and Evaluation
feature_cols = ['temperature', 'promotion', 'month_num'] + region_cols
X = df[feature_cols]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Model Evaluation")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write("Mean Squared Error:", mse)
st.write("R-squared Score:", r2)

st.subheader("Actual vs Predicted Sales (First 50)")
fig6, ax6 = plt.subplots(figsize=(10,5))
ax6.plot(y_test.values[:50], label='Actual')
ax6.plot(y_pred[:50], label='Predicted')
ax6.legend()
st.pyplot(fig6)
