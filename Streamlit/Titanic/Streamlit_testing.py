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


st.write("""My first webpage""")
train = st.file_uploader("pick data file - train")
# test=st.file_uploader("pick data file - test")

st.write("## About the Train dataset")
train_data = pd.read_csv(train)
st.write(train_data.head())
st.write(train_data.describe())
sns.distplot(train_data.Age)
st.pyplot()

train_data['Cabin_flag'] = np.where(train_data.Cabin.isnull(), 0, 1)
y_vars = ['Survived']
x_vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin_flag', 'Embarked']

"""Plotting after combining cats"""
try:

    figsize = (40, 45)  # dynamic figure size
    plt.figure(figsize=figsize);

    for j in y_vars:
        for i in x_vars:
            title = ' % ' + j + ' by ' + i
            plt.subplot(round(len(x_vars) / 2), 3, x_vars.index(i) + 1)
            y = round(
                (train_data.groupby(i)[j].sum().round().astype(int).sort_values() / sum(train_data.Survived)) * 100, 2)
            ax = y.plot(kind='bar', title=title)
            plt.title(title, fontsize=30)
            plt.xticks(fontsize=25)
            for k in range(0, len(y)):
                ax.annotate(y.iloc[k], xy=(k, y.iloc[k] / 2), size=20)
            st.pyplot()

except:
    print('')
plt.title(title, fontsize=30)
plt.xticks(fontsize=25)
plt.plot();

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




