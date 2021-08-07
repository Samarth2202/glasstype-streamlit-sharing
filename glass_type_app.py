# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model,RI, Na, Mg, Al, Si, K, Ca, Ba,Fe):
  glass_type = model.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
  glass_type = glass_type[0]
  if glass_type==1:
    return 'building windows float processed'.upper()
  elif glass_type==2:
    return 'building windows non float processed'.upper()
  elif glass_type==3:
    return 'vehicle windows float processed'.upper()
  elif glass_type==4:
    return 'vehicle windows non float processed'.upper()
  elif glass_type==5:
    return 'containers'.upper()
  elif glass_type==6:
    return 'tableware'.upper()
  else:
    return 'headlamps'.upper()

st.title('Glass Type Predictor')
st.sidebar.title('Exploratory Data Analysis')

if st.sidebar.checkbox('Show raw data'):
  st.subheader('Full Dataset')
  st.dataframe(glass_df)

st.sidebar.subheader('Scatter Plot')
features_list = st.sidebar.multiselect('Select the x-axis values:',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

st.set_option('deprecation.showPyplotGlobalUse', False)
for i in features_list:
  st.subheader(f'Scatter Plot between {i} and Glass Type')
  plt.figure(figsize=(12,6))
  sns.scatterplot(x=i, y='GlassType',data=glass_df)
  st.pyplot()

# Add a subheader in the sidebar with label "Visualisation Selector"
st.sidebar.subheader('Visualisation Selector')
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# and with 6 options passed as a tuple ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot').
# Store the current value of this widget in a variable 'plot_types'.
plot_types = st.sidebar.multiselect('Select the Charts/Plots : ',('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))

if 'Histogram' in plot_types:
  st.subheader('Histogram')
  columns = st.sidebar.selectbox('Select the column to create its histogram : ',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize=(12,6))
  plt.title(f'Histogram for {columns}')
  plt.hist(glass_df[columns],bins='sturges',edgecolor='k')
  st.pyplot()

if 'Box Plot' in plot_types:
  st.subheader('Box Plot')
  columns = st.sidebar.selectbox('Select the column to create its Box Plot : ',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','GlassType'))
  plt.figure(figsize=(12,2))
  plt.title(f'Histogram for {columns}')
  sns.boxplot(glass_df[columns])
  st.pyplot()

if 'Count Plot' in plot_types:
  st.subheader('Count Plot')
  plt.figure(figsize=(12,6))
  sns.countplot(x='GlassType',data=glass_df)
  st.pyplot()
# Create pie chart using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.   
if 'Pie Chart' in plot_types:
  st.subheader('Pie Chart')
  pie_data = glass_df['GlassType'].value_counts()
  plt.figure(figsize=(5,5))
  plt.pie(pie_data, labels=pie_data.index,autopct='%1.2f%%',startangle = 30, explode=np.linspace(.06,.16,6) )
  st.pyplot()
# Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
if 'Correlation Heatmap' in plot_types:
  st.subheader('Correlation Heatmap')
  plt.figure(figsize=(12,6))
  ax = sns.heatmap(glass_df.corr(), annot=True)
  bottom, top = ax.get_ylim()
  ax.set_ylim(bottom + 0.5, top - 0.5)
  st.pyplot()

# Display pair plots using the the 'seaborn' module and the 'st.pyplot()' function. 
if 'Pair Plots' in plot_types:
  st.subheader('Pair Plots')
  plt.figure(figsize=(15,15))
  sns.pairplot(glass_df)
  plt.pyplot()


