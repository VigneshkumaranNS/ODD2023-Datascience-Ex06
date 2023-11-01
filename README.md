### Datascience-Ex06 FEATURE TRANSFORMATION
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
Step1: Read the given Data.
Step2: Clean the Data Set using Data Cleaning Process.
Step3: Apply Feature Transformation techniques to all the features of the data set.
Step4: Print the transformed features.

## Importing libraries and reading csv file:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```

### Basic Information:
```
df.head()
df.info()
```
![277878594-01c70133-1ac6-4ae9-b95d-dfead150c640](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/05d09d18-0377-4fc3-a1ec-f2aba6b026d7)
![2](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/28cdf8ed-f71b-44dd-ba4b-3cade6b97a43)
![3](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/d23977ae-d7ff-4b55-bb0a-585ba3f36e4c)

### Before Transformation:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![1](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/a4631303-b3a7-47bf-b098-385b929911b9)
![2](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/20052fa4-fc16-41db-8f94-a5b8221e04f1)
![3](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/4d1384ce-e56d-4e9d-8807-9ba176815b7e)
![4](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/e5159da8-caa5-435b-b717-7de5adbcf543)



### Log Transformation:
```
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![1](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/7dcfe11a-f40d-424c-b321-b41a4cbea327)
![2](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/1c2d4ba3-632a-4849-b2da-612ecc9c6d88)


### Reciprocal Transformation:
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![3](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/65574444-c672-40bb-a799-56b7345240be)

### SquareRoot Transformation:
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![4](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/4f4ab4d0-7e70-43ba-aef9-9c5c6cac3092)

### Power Transformation:
```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![5](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/69fce838-4d21-41f6-9fa9-8f259beafc65)
![6](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/7a12c932-2cc5-4e6e-8aef-747fd7f3c810)


### Quantile Transormation:
```
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()
```
![7](https://github.com/Adhithyaram29D/ODD2023-Datascience-Ex06/assets/119393540/aba8ff31-0f6c-4e88-a510-6433c1ca94f1)



## Result:
Thus feature transformation is done for the given dataset.
