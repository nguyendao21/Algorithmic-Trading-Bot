# Algorithmic-Trading-Bot

Enhance the existing trading signals with machine learning algorithms that can adapt to new data. This application is the combination of algorithmic trading skills with exisiting skills in financial Python programming and Machine Learning to create algorithmic trading bot that learns and adapts to new data.

The steps behind the application is followed by: 

- Implement an algorithmic trading strategy that uses machine learning to automate the trade decisions.
     - Import the OHLCV dataset into a Pandas DataFrame.
     - Generate trading signals using short- and long-window SMA values.

     - Split the data into training and testing datasets.

     - Use the SVC classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.

     - Review the classification report associated with the SVC model predictions.

     - Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.

     -Create a cumulative return plot that shows the actual returns vs. the strategy returns.his will serve as a baseline against which to compare the effects of tuning the trading algorithm.

- Adjust the input parameters to optimize the trading algorithm.
    - Tune the training algorithm by adjusting the size of the training dataset by sliding the data into different periods.
    - Tune the trading algorithm by adjusting the SMA input features
 
- Evaluate a New Machine Learning Classifier
    - Import a new classifier ( ```LogisticRegression```)
    - Using the original training data as the baseline model, fit another model with the new classifier
    - Backtest the new model to evaluate its performance


---

## Technologies

You’ll use [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/) and the following  **[python version 3.8.5](https://www.python.org/downloads/)** libraries:


* [pandas](https://pandas.pydata.org/docs/)
    * [tseries offsets](https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.DateOffset.html)

* [scikit-learn](https://scikit-learn.org/stable/)
    * [scikit metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) 
    *  [preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
    *  [linear model](https://scikit-learn.org/stable/modules/linear_model.html)

---

## Installation Guide

 ### To check that scikit-learn and hvPlot are installed in your Conda dev environment, complete the following steps:

  ### 1. Activate your Conda dev environment (if it isn’t already) by running the following in your terminal:
```
conda activate dev
```
### 2. When the environment is active, run the following in your terminal to check if the scikit-learn and imbalance-learn libraries are installed on your machine:
```
conda list scikit-learn

```
### If you see scikit-learn and imbalance-learn listed in the terminal, you’re all set!

  ### 1. Install scikit-learn
```
pip install -U scikit-learn
```


---



## Usage

To use this application, simply clone the repository and open jupyter lab from git bash by running the following command:

```jupyter lab```

After launching the application, navigate ``machine_learning_trading_bot.ipynb``

Then in your Jupyter notebook, import the required libraries and dependencies.

```
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
```

## Example
-**Classfication report of SVC model predictions between baseline model (above) and tuned model (below)**

![base_report](https://user-images.githubusercontent.com/94591580/158091056-78863927-b284-4868-ba67-69d210534516.png)![new_ml_report](https://user-images.githubusercontent.com/94591580/158091058-7e1c976f-6f2c-4db2-b76d-6e1f8313827d.png)

The precision measurement between two model looks identical, there is slightly 0.1 value different for value (-1). The recall value of value -1 in tune model (0.33) is higher than the baseline model (0.04) but it still seems very low; The recall value of value 1 in tune model (0.66) is lower than base model (0.96). Overall, it is difficult to tell which model perfomms better.

---

-**Cumulative return plot shows "Actual Returns" and "Strategy Returns" of SVC model predictions between baseline model (above) and tuned model (below)**


![base_plot](https://user-images.githubusercontent.com/94591580/158092214-0a55e558-74d6-4c5e-bc4c-197de26cee7f.png)
![tune_plot](https://user-images.githubusercontent.com/94591580/158092365-1251bfb2-9813-4c1a-b8aa-3e3b7534def5.png)

By contrast, the Strategy Returns overperfom Actual Returns in the Baseline SVM model while the Actual Returns overperform Strategy Returns in Tune SVM model

---

-**New Machine Learning (LogisticRegression) classification report and Cumulative return plot**


![new_ml_plot](https://user-images.githubusercontent.com/94591580/158092908-8829f017-14cb-4624-8c68-354e90a6524a.png)
![new_ml_report](https://user-images.githubusercontent.com/94591580/158092909-1a1f428d-b11e-431d-be70-c0ea8bfc00be.png)

Looking at the plot, the Linear model performs well untill the mid of 2019, that is when actual and predicted return start to differ. To truly find out how well this model works, however, we need to fit it to different sets of pricing data, have it make predictions, backtest it, and then evaluate it against the actual performance of the asset with that trading strategy

### Report

By comparing the SVM model with the Linear model, it shows that the Cumulative Return plot of the SVM model performed better than the Linear model. The Actual Returns seems very identical with the Strategy Return in the SVM model while there are fluctuations in Linear model. However, the classification report of the two model does not have much different, both will gave you about approximately 55% win rate. 



## Contributors

[Nguyen Dao](https://www.linkedin.com/in/nguyen-dao-a55669215/)

daosynguyen21@gmail.com


---

## License

MIT
