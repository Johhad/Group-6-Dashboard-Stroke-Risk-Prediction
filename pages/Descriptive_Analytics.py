import streamlit as st


#🔗 Link: <https://plotly.com/python/scientific-charts/>

@st.cache_data
def load_data():
    import pandas as pd
    df = pd.read_csv('./jupyter-notebooks/processed_data.csv')
    return df

df = load_data()

# Percentage of women
percentage_women = (df['Sex'] =='Female').mean() * 100

# Percentage of men
percentage_men = (df['Sex'] =='Male').mean() * 100

#Mean age 
mean_age = df["Age"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Percentage of women (%)", f"{percentage_women: .2f}", border=True)
col2.metric("Percentage of men (%)", f"{percentage_men: .2f}", border=True)
col2.metric("Mean age (%)", f"{mean_age: .2f}", border=True)

import numpy as np
import plotly.figure_factory as ff

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)


## Plot two

import plotly.express as px
import pandas as pd
df = pd.DataFrame(dict(
    r=[1, 5, 2, 2, 3],
    theta=['processing cost','mechanical properties','chemical stability',
           'thermal stability', 'device integration']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)

st.plotly_chart(fig, use_container_width=True)