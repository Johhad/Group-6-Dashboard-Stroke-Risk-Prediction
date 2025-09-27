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


st.markdown("## Descriptive Analytics")
st.markdown("This page shows key analysis about the dataset we trained")
    
# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Demographics", "Risk Factors", "Correlations"])
    
with tab1:
    col1, col2 = st.columns(2)
        
with col1:
    # Age distribution
    fig_age = px.histogram(data, x='age', nbins=20, title='Age Distribution',
                                   color_discrete_sequence=['#1f77b4'])
    fig_age.update_layout(height=400)
    st.plotly_chart(fig_age, use_container_width=True)
        
with col2:
    # Gender distribution
    gender_counts = data['gender'].value_counts()
    fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                               title='Gender Distribution', color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'})
    fig_gender.update_layout(height=400)
    st.plotly_chart(fig_gender, use_container_width=True)
    
with tab2:
    col1, col2 = st.columns(2)
        
with col1:
    # Stroke distribution
    stroke_counts = data['stroke'].value_counts()
    fig_stroke = px.pie(values=stroke_counts.values, 
                        names=['No Stroke', 'Stroke'],
                        title='Stroke Distribution',
                        color_discrete_sequence=['#2ca02c', '#d62728'])
    fig_stroke.update_layout(height=400)
    st.plotly_chart(fig_stroke, use_container_width=True)
        
with col2:
    # Heart Disease and Stroke
    fig_heart = px.bar(data.groupby(['heart_disease', 'stroke']).size().reset_index(name='count'),
                        x='heart_disease', y='count', color='stroke',
                        title='Heart Disease and Stroke',
                        labels={'heart_disease': 'Heart Disease', 'count': 'Count'})
    fig_heart.update_layout(height=400)
    st.plotly_chart(fig_heart, use_container_width=True)
    
with tab3:
    # Correlation matrix for numerical features
    numerical_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
    corr_matrix = data[numerical_cols].corr()
        
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title='Correlation Matrix of Risk Factors',
                           color_continuous_scale='RdBu')
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)