## imdb_score_prediction

This is a revision project.  
The purpose is to check our knowledges in machine learning.  
For that we import a imdb_score dataset with 5000 entries.

## Data Analysis
For the data analysis we do some cleaning and data visualisation.  
Here are some of the useful libraries used.  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
```

## Data visualization example
![image](https://user-images.githubusercontent.com/57437129/78703460-7ce15f00-790a-11ea-9743-aae3dc8b3e96.png)

## Preprocessing and training
Here are libraries used to train the model
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
```
