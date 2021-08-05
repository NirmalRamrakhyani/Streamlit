#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import io

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette('Set2')
from sklearn.model_selection import train_test_split

# In[12]:


st.write("""# The EDA Shop
(Work in Progress)
### By [Nirmal Ramrakhyani](https://www.linkedin.com/in/nirmal-ramrakhyani-32993a101/)
""")
train = st.file_uploader("Upload the data here!")
@st.cache
def get_data():
    # test=st.file_uploader("pick data file - test")
    return pd.read_csv(train)

train = get_data()

st.write("## About the dataset")
st.write("### Top 5 rows ")

st.dataframe(train.head().style)
st.write("### Basic Stats ")
st.write(train.describe())

st.write("### Missing Value Analysis ")
#train
try :
    plt.figure(figsize=(10,10))
#     plt.subplot(1,2,1)
    y=train.isna().sum().loc[(train.isna().sum().sort_values(ascending=False))>0].apply(lambda x:x/len(train)).sort_values(ascending=False)
    y.plot(kind='bar')
    title = '% Missing Values by fields in Train'
    ax=y.plot(kind='bar',title=title)
    plt.title(title,fontsize=20)
    plt.xticks(fontsize=15)
    for i in range(0,len(y)):
        ax.annotate(round(y.iloc[i,]*100,1), xy =(i, y.iloc[i]/2),size=15)
    st.pyplot()

#     plt.subplot(1,2,2)
#     y=test.isna().sum().loc[(test.isna().sum().sort_values(ascending=False))>0].apply(lambda x:x/len(test)).sort_values(ascending=False)
#     y.plot(kind='bar')
#     title = '% Missing Values by fields in Train'
#     ax=y.plot(kind='bar',title=title)
#     for i in range(0,len(y)):
#         ax.annotate(round(y.iloc[i,]*100,1), xy =(i, y.iloc[i]/2),size=14)
except:
    print('')
plt.title(title,fontsize=20)
plt.xticks(fontsize=15)
plt.plot();

st.write("### Distribution Plots ")
sns.distplot(train.Age)
st.pyplot();

# st.write("### Select X and Y variables ")
train['Cabin_flag'] = np.where(train.Cabin.isnull(), 0, 1)


# y_vars = st.radio("Select Target variable",tuple(train.columns))
y_vars = ['Survived']
x_vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin_flag', 'Embarked']
# x_vars = st.multiselect("Select X variables",tuple(x_vars))

st.write("## X variables vs Y")
try:

    figsize = (40, 45)  # dynamic figure size
    plt.figure(figsize=figsize);

    for j in y_vars:
        for i in x_vars:
            title = ' % ' + j + ' by ' + i
            plt.subplot(round(len(x_vars) / 2), 3, x_vars.index(i) + 1)
            y = round(
                (train.groupby(i)[j].sum().round().astype(int).sort_values() / sum(train.Survived)) * 100, 2)
            ax = y.plot(kind='bar', title=title)
            plt.title(title, fontsize=20)
            plt.xticks(fontsize=15)
            for k in range(0, len(y)):
                ax.annotate(y.iloc[k], xy=(k, y.iloc[k] / 2), size=15)
            st.pyplot()

except:
    print('')
plt.title(title, fontsize=10)
plt.xticks(fontsize=10)


# multiple_files = st.file_uploader(
#     "Multiple File Uploader",
#     accept_multiple_files=True
# )
# for file in multiple_files:
#     dataframe = pd.read_csv(file)
#     file.seek(0)
#     st.write(dataframe)

# st.write(dataframe.head())

# In[ ]:




