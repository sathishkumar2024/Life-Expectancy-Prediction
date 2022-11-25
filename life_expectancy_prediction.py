import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

## we load the data


st.title('Life Expectancy Prediction')
st.write('**description of data**')
st.write('COUNTRY: it gives the country names')
st.write('YEAR: it give the year')
st.write('STATUS:Country Developed or Developing status')
st.write('Adult Mortality: Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)')
st.write('infant deaths: Number of Infant Deaths per 1000 population')
st.write('Alcohol: Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol) -percentage expenditure: Expenditure on health as a percentage of Gross Domestic Product per capita(%)')
st.write('Hepatitis B: Hepatitis B (HepB) immunization coverage among 1-year-olds (%)')
st.write('Measles: Measles - number of reported cases per 1000 population')
st.write('BMI: Average Body Mass Index of entire population')
st.write('under-five deaths: Number of under-five deaths per 1000 population')
st.write('Polio: Polio (Pol3) immunization coverage among 1-year-olds (%)')
st.write('Total expenditure: General government expenditure on health as a percentage of total government expenditure (%)')
st.write('Diphtheria: Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)')
st.write('HIV/AIDS: Deaths per 1 000 live births HIV/AIDS (0-4 years)')
st.write('GDP: Gross Domestic Product per capita (in USD)')
st.write('Population: Population of the country')
st.write('thinness 1-19 years: Prevalence of thinness among children and adolescents for Age 10 to 19 (%)')
st.write('thinness 5-9 years: Prevalence of thinness among children for Age 5 to 9(%)')
st.write('Income composition of resources: Human Development Index in terms of income composition of resources (index ranging from 0 to 1)')
st.write('Schooling: Number of years of Schooling(years)')


# load the data set
st.write('**sample data**')
data = pd.read_csv('C:/Users/Sathish/skilvertax/sample.csv', encoding='utf-8')

y = data['life_expectancy']
X = data.drop(['country','status','life_expectancy','under-five_deaths','year'],axis = 1)
st.write(data.head(10))

## visualization
chart_select = st.sidebar.selectbox(
    label='Select the type of chart',
    options=['scatterplot','boxplot','regression_plot','distribution_plot','histogram']
)

numeric_columns = list(data.select_dtypes(['float','int']).columns)

if chart_select == 'scatterplot':
    st.sidebar.subheader('choose the features')
    x_values = st.sidebar.selectbox('x axis',options= numeric_columns)
    y_values = st.sidebar.selectbox('y axis',options=numeric_columns)
    plot = plt.scatter(data = data,x=x_values,y= y_values)
    st.write(plot)
if chart_select == 'boxplot':
    st.sidebar.subheader('choose the features')
    y_values = st.sidebar.selectbox('y axis',options=numeric_columns)
    plot = sns.boxplot(data = data,y= y_values)
    st.write(plot)
if chart_select == 'regression_plot':
    st.sidebar.subheader('choose the features')
    x_values = st.sidebar.selectbox('x axis',options= numeric_columns)
    y_values = st.sidebar.selectbox('y axis',options=numeric_columns)
    plot = sns.regplot(data = data,x=x_values,y= y_values)
    st.write(plot)
if chart_select == 'distribution_plot':
    st.sidebar.subheader('choose the features')
    x_values = st.sidebar.selectbox('x axis',options= numeric_columns)
    y_values = st.sidebar.selectbox('y axis',options=numeric_columns)
    plot = sns.distplot(data = data,x=x_values,y= y_values)
    st.write(plot)
if chart_select == 'histogram':
    st.sidebar.subheader('choose the features')
    x_values = st.sidebar.selectbox('x axis',options= numeric_columns)
    plot = plt.hist(data = data,x=x_values,)
    st.write(plot)

# define the user inputs

st.sidebar.header('User Input Parameters')

def user_input_features():
    adult_mortality = st.sidebar.slider('Adult_Mortality',float(X.adult_mortality.min()),float(X.adult_mortality.max()),float(X.adult_mortality.mean()))
    Infant_Deaths = st.sidebar.slider('Infant_Deaths',float(X.infant_deaths.min()),float(X.infant_deaths.max()),float(X.infant_deaths.mean()))
    Alchohol = st.sidebar.slider('Alchohol',float(X.alchohol.min()),float(X.alchohol.max()),float(X.alchohol.mean()))
    Percentage_Expenditure = st.sidebar.slider('Percentage_Expenditure',float(X.percentage_expenditure.min()),float(X.percentage_expenditure.max()),float(X.percentage_expenditure.mean()))
    Hepatitis_b = st.sidebar.slider('Hepatitis_b',float(X.hepatitis_b.min()),float(X.hepatitis_b.max()),float(X.hepatitis_b.mean()))
    Measles = st.sidebar.slider('Measles',float(X.measles.min()),float(X.measles.max()),float(X.measles.mean()))
    BMI = st.sidebar.slider('BMI',float(X.bmi.min()),float(X.bmi.max()),float(X.bmi.mean()))
    Polio = st.sidebar.slider('Polio',float(X.polio.min()),float(X.polio.max()),float(X.polio.mean()))
    Total_Expenditure = st.sidebar.slider('Total_Expenditure',float(X.total_expenditure.min()),float(X.total_expenditure.max()),float(X.total_expenditure.mean()))
    Diphtheria = st.sidebar.slider('Diphtheria',float(X.diphtheria.min()),float(X.diphtheria.max()),float(X.diphtheria.mean()))
    AIDS = st.sidebar.slider('AIDS',float(X['hiv/aids'].min()),float(X['hiv/aids'].max()),float(X['hiv/aids'].mean()))
    GDP = st.sidebar.slider('GDP',float(X.gdp.min()),float(X.gdp.max()),float(X.gdp.mean()))
    Thinness_1_19_Years = st.sidebar.slider('Thinness_1_19_Years',float(X.thinness_1_19_years.min()),float(X.thinness_1_19_years.max()),float(X.thinness_1_19_years.mean()))
    Thinness_5_9_Years = st.sidebar.slider('Thinness_5_9_Years',float(X.thinness_5_9_years.min()),float(X.thinness_5_9_years.max()),float(X.thinness_5_9_years.mean()))
    Income_Composition_Of_Resources = st.sidebar.slider('Income_Composition_Of_Resources',float(X.income_composition_of_resources.min()),float(X.income_composition_of_resources.max()),float(X.income_composition_of_resources.mean()))
    Schooling = st.sidebar.slider('Schooling',float(X.schooling.min()),float(X.schooling.max()),float(X.schooling.mean()))
    data= {
            'adult_mortality':adult_mortality,
            'Infant_deaths':Infant_Deaths,
            'Alchohol':Alchohol,
            'Percentage_Expenditure':Percentage_Expenditure,
            'Hepatitis_b':Hepatitis_b,
            'Measles':Measles,
            'BMI':BMI,
            'Polio':Polio,
            'Total_Expenditure':Total_Expenditure,
            'Diphtheria':Diphtheria,
            'AIDS':AIDS,
            'GDP':GDP,
            'Thinness_1_19_Years':Thinness_1_19_Years,
            'Thinness_5_9_Years':Thinness_5_9_Years,
            'Income_Composition_Of_Resources':Income_Composition_Of_Resources,
            'Schooling':Schooling}

    features = pd.DataFrame(data,index = [0])
    return features

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# fit the model
model = pickle.load(open('C:/Users/Sathish/skilvertax/life_expectancy_trained_data.sav', 'rb'))
#model = GradientBoostingRegressor()
gbr_model = model
gbr_model.fit(X,y)

pred = gbr_model.predict(df)

st.header('prediction of Life Expectancy')
st.write(pred)







