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
income["target"] = np.where(income["SalStat"] == ' less than or equal to 50,000', 0, 1)

income["target"].value_counts(normalize=True).plot.bar()
````

3. Explore the numeric features using histograms or boxplots.
4. Explore the categorical features using bar charts.
5. Consider writing functions for steps 3 and 4.

























