# Zomato_Analysis

This project aims to analyze data from the Zomato restaurant database to uncover patterns and insights about various restaurants. The analysis involves cleaning and processing the data, handling missing values, and performing exploratory data analysis (EDA).

<p>

---

## Project Overview

The Zomato Analysis project involves several key steps:

1. **Data Loading and Initial Exploration**:
    - The dataset is loaded using `pandas` and the initial structure and columns are examined to understand its content.
    - Basic statistics such as shape and count of missing values are computed to assess the data quality.

2. **Data Cleaning**:
    - Columns with missing values are identified and the percentage of missing values is calculated.
    - The `rate` column is cleaned by removing unwanted characters and converting it to a numeric type.
    - Rows with missing ratings are removed to ensure accurate analysis.

3. **Calculating Average Ratings**:
    - The average rating for each restaurant is calculated using a pivot table.
    - Restaurants are then sorted by their average rating to identify the top-rated establishments.

4. **Visualizing Rating Distribution**:
    - A distribution plot is created to visualize the spread of ratings across all restaurants.
    - Statistical tests are performed to check the normality of the rating distribution.

5. **Analyzing Restaurant Chains**:
    - The most common restaurant chains are identified and visualized using a bar plot.
    - The acceptance of online orders and table bookings by restaurants is analyzed using pie charts.

6. **Restaurant Type Analysis**:
    - The distribution of different types of restaurants is analyzed and visualized.
    - The average number of votes received by restaurants is calculated and visualized.

7. **Top and Worst Restaurants**:
    - The highest-rated restaurants based on the number of votes are identified and visualized.
    - Similarly, the worst-rated restaurants with a significant number of votes are also identified and visualized.

8. **Location and Cuisine Analysis**:
    - The distribution of restaurants across different locations is visualized.
    - The top 10 cuisines offered by restaurants are identified and visualized.
    - The distribution of the cost for two people is analyzed.
  
## Download dataset
    - https://drive.google.com/file/d/1O9HKIWj-4G2J1D_5VbAFtfLoxN040ZBo/view?usp=sharing

## Code Breakdown

### Data Loading and Initial Cleaning
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("/content/drive/MyDrive/Python_Session/zomato (1).csv")
```

### Handling Missing Values
```python
feature_na = [i for i in data.columns if data[i].isnull().sum() > 0]
for i in feature_na:
    print(f"{i} has {np.round((data[i].isnull().sum()/len(data[i])*100), 4)}% null values")
```

### Data Transformation
```python
data.dropna(subset=['rate'], axis=0, inplace=True)
def split(x):
    return x.split('/')[0].strip()
data['rate'] = data['rate'].apply(split)
data['rate'].replace(['NEW', '-'], 0, inplace=True)
data['rate'] = data['rate'].astype(float)
```

### Exploratory Data Analysis
```python
# Average rating calculation
rating = pd.pivot_table(data, index='name', values='rate').sort_values('rate', ascending=False)
plt.figure(figsize=(15,8))
sns.barplot(x=rating[0:20].rate, y=rating[0:20].index, orient="h")
plt.show()

# Rating distribution
sns.set_style('whitegrid')
sns.distplot(data['rate'])
plt.show()

# Restaurant chains analysis
plt.figure(figsize=(10,7), dpi=110)
chains = data['name'].value_counts()[0:15]
sns.barplot(x=chains, y=chains.index, palette='deep')
plt.xlabel("Number of outlets")
plt.show()

# Online order and table booking analysis
x = data.online_order.value_counts()
labels = ['accepted', 'not-accepted']
plt.pie(x, labels=labels, explode=[0.0, 0.1], autopct='%1.1f%%')
plt.show()

x = data.book_table.value_counts()
labels = ['accepted', 'not-accepted']
plt.pie(x, labels=labels, explode=[0.0, 0.1], autopct='%1.1f%%')
plt.show()

# Restaurant type analysis
plt.figure(figsize=(20,12))
rest_type = data.rest_type.value_counts()[0:15]
plt.bar(rest_type.index, rest_type)
plt.show()

# Top and worst restaurants analysis
high_vot = voting[voting['votes'] > 5000]
plt.figure(figsize=(20,12), dpi=100)
plt.barh(high_vot.index, high_vot['votes'])
plt.show()

bad_count = voting_rating[(voting_rating['rate'] < 3) & (voting_rating['rate'] > 0)]
bad_restaurants = bad_count[bad_count['votes'] > 500].groupby('name')[['rate']].mean()
plt.figure(figsize=(20,8))
sns.barplot(x=bad_restaurants['rate'], y=bad_restaurants.index)
plt.show()

# Location and cuisine analysis
location_restro = data[['location']].value_counts()[0:20]
plt.figure(figsize=(25,8))
sns.countplot(x=data['location'])
plt.xticks(rotation=90)
plt.show()

cusin = data.cuisines.value_counts()[0:10]
sns.barplot(x=cusin.index, y=cusin)
plt.xticks(rotation=75)
plt.show()

# Cost for two people analysis
data.dropna(axis=0, subset=['approx_cost(for two people)'], inplace=True)
```

## Conclusion

This analysis of the Zomato dataset provides insights into how various factors like ratings, votes, restaurant types, locations, and cuisines impact restaurant popularity and performance. The visualizations help identify trends and patterns that can be useful for restaurant owners and customers.

## Future Work

- Further analysis can be conducted to understand the impact of other factors such as restaurant ambiance and customer reviews.
- Machine learning models can be developed to predict restaurant ratings based on the features available in the dataset.

---

</p>
