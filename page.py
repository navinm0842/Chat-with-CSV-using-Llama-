import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from script import run_llama  # Import the Llama model function
import tempfile
import chardet
from scipy.stats import spearmanr, kendalltau


# Function to display basic statistics
def display_statistics(var1):
    mean = np.mean(var1)
    median = np.median(var1)
    std_dev = np.std(var1)
    
    st.write(f"Mean: {mean}")
    st.write(f"Median: {median}")
    st.write(f"Standard Deviation: {std_dev}")
    
# Function to plot histogram
def plot_histogram(df,column):
    plt.figure(figsize=(10, 6))
    plt.hist(column)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title('Histogram')
    st.pyplot(plt.gcf())

# Function to plot box plot
def plot_boxplot(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    st.pyplot(plt.gcf())

# Function to plot bar plot
def plot_barplot(df, column):
    plt.figure(figsize=(10, 6))
    plt.bar(df,column)
    st.pyplot(plt.gcf())

# Function to plot scatter plot
def plot_scatterplot(df, col1, col2):
    plt.figure(figsize=(10, 6))
    plt.scatter(df(col1), df(col2))
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Scatter Plot')
    st.pyplot(plt.gcf())

# Function to plot heatmap
def plot_heatmap(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    st.pyplot(plt.gcf())

# Streamlit app
st.title("👨‍💻 Chat with your CSV")
st.title('CSV File Visualizer and Analyzer with Llama-2 Model')
# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file",type="csv")

if uploaded_file is not None:
    st.write("filename:",uploaded_file.name)
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    # Options to select columns
    type_visual=['Histogram','Box Plot','Scatter plot','Bar chart','Heatmap','Statistics']
   
    columns = df.columns.tolist()
    
    query = st.text_area("Insert your Query")
    button_1 = st.button(key='prompt',label="Generate") 
    if button_1 and query:
        st.write('Loading....')
        result = run_llama(query,uploaded_file)
        st.write(result)

    #column1 = st.sidebar.selectbox("Select First Column for Scatter Plot", columns, index=0)
    #column2 = st.sidebar.selectbox("Select Second Column for Scatter Plot", columns, index=1)
    button = st.button('Click for Visualization')

    if button:
        st.sidebar.header('Visualization Options')
        selected_column = st.sidebar.selectbox("Select Column for Visualization", type_visual)
        while selected_column:
            if selected_column=='Histogram':
                h1 = st.sidebar.selectbox("Select Column for Scatter Plot", columns)
                plot_histogram(df,h1)
            elif selected_column=='Bar chart':
                b1 = st.sidebar.selectbox("Select First Column for Bar Chart", columns)
                b2 = st.sidebar.selectbox("Select Second Column for Bar chart", columns)
                plot_barplot(b1,b2)
            elif selected_column=='Scatter Plot':
                s1 = st.sidebar.selectbox("Select First Column for Scatter Plot", columns)
                s2 = st.sidebar.selectbox("Select Second Column for Scatter Plot", columns)
                plot_scatterplot(s1, s2)
            elif selected_column=='Bar Chart':
                bc1 = st.sidebar.selectbox("Select First Column for Box Plot", columns)
                bc2 = st.sidebar.selectbox("Select Second Column for Box Plot", columns)
                plot_boxplot(bc1, bc2)
            elif selected_column=='Heatmap':
                hm1 = st.sidebar.selectbox("Select Column for Heatmap", columns)
                plot_heatmap(hm1)
            elif selected_column=='Statistics':
                column1 = st.sidebar.selectbox("Select Main Column for Statistics", columns)
                display_statistics(column1)