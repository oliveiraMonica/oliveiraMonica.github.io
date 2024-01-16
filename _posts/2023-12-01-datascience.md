---
title: EDA for Classification
author: Oliveira Monica
date: 2023-12-01 16:25:00 -0300
categories: [Data Science, Pre-Modeling Prep & EDA]
tags: [blog]
---

Exploratory Data Analysis, or EDA for short, is the process of exploring and visualizing data to find useful patterns and insights that help inform the modeling process.
Oftentimes when we're working with classification data, we might have dozens, if not hundreds of potential features at our disposal. <br>
So the process of EDA is largely about identifying which features are most promising and narrowing down to just a handful, at least to start, as we build a baseline model, we can always add more features later.

<br>
When performing EDA for classification, it`s important to explore: <br>
- The target variale  <br>
- The features  <br>
- Feature-target relationships  <br>
- Feature-feature relationships  <br>

## Project 1 - income prediction project <br>
We want to be able to validate whether our customers are reporting income accurately using machine learning.

### Assignment 1: EDA

1. Read in `income.csv`

````python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read file
income = pd.read_csv("Data/income.csv")
income.head()
````
![head](/assets/img_data_science/income-read.png)



2. Convert the target, `SalStat` into a binary numeric variable called `target`, and build a bar chart that plots the frequency of each value.

````python
# Convert Salt
income["target"] = np.where(income["SalStat"] == ' less than or equal to 50,000', 0, 1)

# Plot
income["target"].value_counts(normalize=True).plot.bar()
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Proportion')
plt.show()
````
![head](/assets/img_data_science/income-ploty.png){: width="500"}

3. Explore the numeric features using histograms or boxplots.


````python
# 
def num_box_plotter(data):
  for column in data.select_dtypes("number"):
  sns.boxplot(data[column]).set(ylabel=column)
  plt.show()

num_box_plotter(income)

````
![head](/assets/img_data_science/income-fig1.png){: width="500"}
![head](/assets/img_data_science/income-fig2.png){: width="500"}
![head](/assets/img_data_science/income-fig3.png){: width="500"}
![head](/assets/img_data_science/income-fig4.png){: width="500"}
![head](/assets/img_data_science/income-fig5.png){: width="500"}

4. Explore the categorical features using bar charts.

````python
# 
def cat_bar_plotter(data, normalize=False):

    for column in data.select_dtypes("object"):
        data[column].value_counts(normalize=normalize).plot.bar()
        plt.show()

cat_bar_plotter(income, normalize=True)
````
![head](/assets/img_data_science/income-categorical-fig1.png){: width="500"}
![head](/assets/img_data_science/income-categorical-fig2.png){: width="500"}
![head](/assets/img_data_science/income-categorical-fig3.png){: width="500"}
![head](/assets/img_data_science/income-categorical-fig4.png){: width="500"}
![head](/assets/img_data_science/income-categorical-fig5.png){: width="500"}
![head](/assets/img_data_science/income-categorical-fig6.png){: width="500"}
![head](/assets/img_data_science/income-categorical-fig7.png){: width="500"}
![head](/assets/img_data_science/income-categorical-fig8.png){: width="500"}
![head](/assets/img_data_science/income-categorical-fig9.png){: width="500"}


[Code Available Here](https://github.com/oliveiraMonica/data_science)

[Link Data file](https://drive.google.com/file/d/1jKoH0EfgDoSU4C0yCJOJT5cVtln_3T2c/view?usp=drive_link)



























