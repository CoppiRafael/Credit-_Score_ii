#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:#8f28ff; color:white; padding:10px; border-radius:5px; text-align:center; font-size:20px;">
#     Credit Scoring Analysis and Prediction: A Study Based on CRISP-DM
# </div>

# **Project Description:**
# 
# > This project aims to develop a comprehensive credit scoring prediction system using the CRISP-DM (Cross Industry Standard Process for Data Mining) framework. The analysis involves an extensive range of approaches, including multiple machine learning algorithms, automated ML with PyCaret, statistical tests, and iterative modeling to evaluate different techniques. The dataset contains financial and behavioral client information, such as credit history, income, occupation, and payment behavior. The project will cover data exploration, preprocessing, predictive model development, evaluation, and result interpretation to offer actionable insights and ensure a robust and versatile solution for financial decision-making.
# 

# ![image.png](attachment:15574609-bf6f-4218-a9fe-8b1c45a0574a.png)

# <div style="background-color:#8f28ff; color:white; padding:10px; border-radius:5px; text-align:center; font-size:20px;">
#     Business Understanding
# </div>

# ## Objective
# The primary objective of this project is to develop a comprehensive and reliable credit scoring prediction system. This system aims to support financial institutions in assessing the creditworthiness of potential clients by leveraging advanced data analytics and machine learning methodologies.
# 
# ## Business Goals
# 1. **Risk Mitigation:**
#    - Reduce the risk of granting loans to clients who may default on payments.
#    - Improve the accuracy of credit scoring to minimize financial losses.
# 
# 2. **Client Segmentation:**
#    - Identify different client profiles based on their credit behavior and financial attributes.
#    - Tailor financial products and services to specific customer segments.
# 
# 3. **Decision Support:**
#    - Provide actionable insights to assist decision-makers in loan approval processes.
#    - Ensure compliance with regulatory standards in credit risk assessment.
# 
# 4. **Optimization:**
#    - Enhance the efficiency of the credit scoring process by automating repetitive tasks.
#    - Reduce the time and resources required for manual credit assessment.
# 
# ## Key Questions
# - What are the main factors influencing a client’s creditworthiness?
# - How can machine learning models be used to predict the likelihood of a client defaulting?
# - What is the financial impact of improving the accuracy of credit scoring models?
# - Which statistical and machine learning techniques provide the best performance for this problem?
# 
# ## Constraints
# - **Data Quality:**
#   - Ensure data completeness and accuracy to build robust models.
#   - Address potential issues such as missing values and outliers.
# 
# - **Regulatory Compliance:**
#   - Adhere to financial regulations and ethical considerations in data usage.
# 
# - **Performance Metrics:**
#   - Balance the trade-off between precision and recall to meet business needs.
# 
# ## Deliverables
# - A well-documented credit scoring system that integrates:
#   - Multiple machine learning algorithms.
#   - Automated ML (AutoML) workflows using PyCaret.
#   - Statistical tests and hypothesis validation.
# 
# - Comprehensive insights into client credit behavior and risk profiles.
# - Visualizations and reports to support decision-making processes.
# 
# 

# <div style="background-color:#8f28ff; color:white; padding:10px; border-radius:5px; text-align:center; font-size:20px;">
#     Data Understanding
# </div>

# | Variable Name            | Description                                              |
# |-----------------------------|-------------------------------------------------------------|
# | data_ref                   | Represents the reference date of the data.                |
# | index                      | Index of the record in the dataset.                       |
# | sexo                       | Indicates the person's gender.                            |
# | posse_de_veiculo           | Indicates whether the person owns a vehicle.              |
# | posse_de_imovel            | Indicates whether the person owns a property.             |
# | qtd_filhos                 | Represents the number of children the person has.         |
# | tipo_renda                 | Represents the type of income source of the person.       |
# | educacao                   | Represents the person's education level.                  |
# | estado_civil               | Indicates the person's marital status.                    |
# | tipo_residencia            | Represents the type of residence of the person.           |
# | idade                      | Indicates the person's age.                               |
# | tempo_emprego              | Represents the person's employment duration (in years).   |
# | qt_pessoas_residencia      | Represents the number of people living in the residence.  |
# | renda                      | Represents the person's monthly income (in local currency).|
# | mau                        | Indicates whether the person has bad credit behavior.     |
# 
# 

# In[334]:


import warnings
import math

import pandas                  as pd
import numpy                   as np
import matplotlib.pyplot       as plt
import seaborn                 as sns
import plotly.express          as px
import statsmodels.api         as sm
import statsmodels.formula.api as smf
import xgboost                 as xgb
import scipy.stats             as stats



from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn                              import metrics
from scipy.stats                          import ks_2samp
from scipy.stats                          import t
from sklearn.model_selection              import train_test_split
from sklearn.model_selection              import GridSearchCV,RandomizedSearchCV
from sklearn.impute                       import SimpleImputer
from sklearn.metrics                      import accuracy_score, roc_auc_score, classification_report,mean_squared_error
from sklearn.compose                      import ColumnTransformer
from sklearn.pipeline                     import Pipeline
from sklearn.preprocessing                import OneHotEncoder, StandardScaler
from sklearn.decomposition                import PCA
from sklearn.linear_model                 import LogisticRegression
from scipy.stats                          import uniform


# In[335]:


warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.3f}'.format
get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_feather('credit_scoring.ftr')

df_ = df.drop_duplicates()
df_ = df_.sort_values(by='data_ref')

df_out_of_time = df_[df_['data_ref'] >= '2016-01-01']
data_frame = df_[df_['data_ref'] < '2016']

def data_out_of_time() -> pd.DataFrame:
    print("Original data_out_of_time loaded")
    return df_out_of_time.copy()

def data_2015_dev_ml() -> pd.DataFrame:
    print("Original data_2015_dev_ml loaded")
    return data_frame.copy()

def data_without_duplicated() -> pd.DataFrame:
    print("Original data_without_duplicated loaded")
    return df_.copy()

# In[355]:

if __name__ == "__main__":
    df_.data_ref.value_counts().to_frame().sort_values(by='data_ref')
    
    
    # In[356]:
    
    
    
    # In[357]:
    
    
    color_palette = ['#8f28ff','#bc7eff','#c0b0d1', '#a1b5db', '#7095db', '#636efa']
    sns.set_palette(sns.color_palette(color_palette))
    
    
    # In[358]:
    
    
    df_out_of_time.data_ref.value_counts().to_frame().reset_index()
    
    
    # In[359]:
    
    
    fig = px.bar(data_frame = df_out_of_time.data_ref.value_counts().to_frame().reset_index(),x='data_ref',y='count',height=550,text_auto=True,)
    fig.show()
    
    
    # <div style="background-color:#8f28ff; color:white; padding:10px; border-radius:5px; text-align:center; font-size:20px;">
    #     Exploratory Data Analysis
    # </div>
    
    # ## Continuous Features
    
    # In[360]:
    
    
    num_cols = df_.select_dtypes(include='number').columns.drop('index')
    num_plots = len(num_cols)
    
    fig, axes = plt.subplots(nrows=(num_plots // 3) + 1, ncols=3, figsize=(17, 13))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.histplot(data=df_, x=col, hue='mau', bins=30, ax=axes[i],alpha=1,multiple="stack")
        axes[i].set_title(f'Histogram of {col}')
        axes[i].grid(False)
    
    plt.tight_layout()
    plt.show()
    
    
    # #### Insight: 
    # > Analyzing only the distributions of the variables in relation to default, the data does not show us relevant information, notice by the color differences that the pattern that is presented in defaulters also occurs in defaulters, but let's investigate further how these continuous variables relate to other variables, do they influence and help predict something? we will see below
    
    # In[361]:
    
    
    fig = px.histogram(data_frame = df_[df_['renda']<=100000],x='renda',color='mau',template='simple_white',color_discrete_sequence=['#9043e4', '#c0b0d1'])
    fig.show()
    
    
    # In[362]:
    
    
    fig, axes = plt.subplots(nrows=(num_plots // 3) + 1, ncols=3, figsize=(15, 9))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df_, x=col,hue='mau', linewidth=.7, ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].grid(False)
    
    plt.tight_layout()
    plt.show()
    
    
    # #### Insight 
    # > Here we can see the huge number of possible outliers in our data, variables such as income and length of employment need to be treated and standardized in the future, since we saw at the beginning that the standard deviation of these variables is very high.
    # > By analyzing the medians of the graphs and the relationship between the quartiles, we can see that length of employment and income seem to be very interesting for selecting possible outliers; note in the length of employment graph that people with shorter lengths of employment are more likely to be defaulters.
    
    # In[363]:
    
    
    fig = px.scatter_matrix(df_, dimensions=num_cols, color='mau',
                            title='Pair Chart',
                            height = 700,color_discrete_sequence=['#9043e4', '#c0b0d1']
                           )
    
    fig.show()
    
    
    # #### insight:
    # > Although income is very skewed, with a very high standard deviation and very out-of-range values, we can see that in relation to any other numerical feature it always shows the same pattern, defaulting customers are in the 'less income' part. as I said, it needs to be dealt with, but apparently it can be a good variable for the development of our machine learning algorithm. 
    
    # In[364]:
    
    
    figure = px.scatter(data_frame=df_,x='renda',y='tempo_emprego',height = 550,color_discrete_sequence=['#9043e4', '#c0b0d1'],color='mau')
    figure.show()
    
    
    # In[365]:
    
    
    corr = df_.select_dtypes(include='number').corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask,cbar=True,annot=True,vmax=1,vmin=-1, cmap=sns.color_palette(color_palette, as_cmap=True))
    plt.title('Heatmap of Features Correlations')
    plt.grid(None)
    plt.show()
    
    
    # ## Categorical Features
    
    # In[366]:
    
    
    df.info()
    
    
    # In[367]:
    
    
    cat_var = df_.select_dtypes(exclude='number').drop(columns=['mau','data_ref']).columns.tolist()
    
    
    # In[368]:
    
    
    cat_var
    
    
    # In[369]:
    
    
    num_vars = len(cat_var)
    num_cols = 2 if num_vars <= 6 else 3
    num_rows = math.ceil(num_vars / num_cols) 
    
    current_var_index = 0  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    
    if num_rows > 1 and num_cols > 1:
        axes = axes.flatten()
    elif isinstance(axes, plt.Axes): 
        axes = [axes]
    
    for ax in axes:
        if current_var_index < num_vars:
            var = cat_var[current_var_index]
            
            sns.countplot(data=df_, x=var, hue='mau', ax=ax)
            ax.tick_params(axis='x', rotation=90)  
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='baseline',
                            fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')
    
            ax.set_title(f'Countplot de {var}', fontsize=12)
            ax.set_xlabel(var, fontsize=10)
            ax.set_ylabel('Contagem', fontsize=10,)
            
            current_var_index += 1
        else:
            ax.axis('off') 
    
    plt.tight_layout()
    plt.grid(None)
    plt.show()
    
    
    # #### insight
    # > the distribution of categorical variables seems to be quite consistent in terms of frequency
    
    # ## Bivariate
    
    # In[370]:
    
    
    df_.groupby('mau')['idade'].agg(['mean', 'median', 'std'])
    
    
    # In[371]:
    
    
    df_.groupby('mau')['tempo_emprego'].agg(['mean', 'median', 'std'])
    
    
    # #### insight
    # > Higher mean and deviation, demonstrating and confirming what we had already said. length of employment is a good variable to consider
    
    # In[372]:
    
    
    df.groupby('mau')['tempo_emprego'].describe()
    
    
    # In[373]:
    
    
    df_.groupby('mau')['renda'].agg(['mean', 'median', 'std'])
    
    
    # #### insight
    # > income has the same characteristics as length of employment but in much greater proportions, which leads us to understand that it needs very careful treatment. we need to separate it and classify it in a way that does not make its information so discrepant
    
    # In[374]:
    
    
    df.groupby('mau')['renda'].describe()
    
    
    # In[375]:
    
    
    pd.crosstab(df['tipo_renda'], df['mau'], normalize='index')
    
    
    # In[376]:
    
    
    df.pivot_table(index='educacao', columns='mau', values='idade', aggfunc='mean')
    
    
    # In[377]:
    
    
    tab = pd.crosstab(df_['sexo'],df_['mau'])
    tab
    
    
    # In[378]:
    
    
    tab['total'] = tab.sum(axis=1)
    tab
    
    
    # In[379]:
    
    
    dfx = df_.copy()
    dfx['ano'] = dfx['data_ref'].dt.year
    dfx['mes'] = dfx['data_ref'].dt.month
    
    
    # In[380]:
    
    
    dfx['media_cumulativa_renda'] = dfx['renda'].expanding().mean()
    
    
    sns.lineplot(data=dfx, x='data_ref', y='media_cumulativa_renda',hue='mau')
    plt.title('Cumulative Average Income Over Time')
    plt.ylabel('Cumulative Average Income')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.grid(None)
    plt.show()
    
    
    # In[381]:
    
    
    dfx['media_cumulativa_tempo_emprego'] = dfx['tempo_emprego'].expanding().mean()
    
    
    sns.lineplot(data=dfx, x='data_ref', y='media_cumulativa_tempo_emprego',hue='mau')
    plt.title('Cumulative Average Employment Time Over Time')
    plt.ylabel('Cumulative average employment time ')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.grid(None)
    plt.show()
    
    
    # In[382]:
    
    
    inadimplencia_tempo = dfx.groupby(['ano', 'mes'])['mau'].mean().reset_index()
    
    sns.lineplot(data=inadimplencia_tempo, x='mes', y='mau', hue='ano',palette=color_palette)
    plt.title('Default rate over time')
    plt.ylabel('Default rate')
    plt.xlabel('Month')
    plt.grid(None)
    plt.show()
    
    
    # In[383]:
    
    
    num_cols = 2 if num_var < 6 else 3
    num_rows = math.ceil(num_var / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))  
    axes = axes.flatten()  # Achatar o array de eixos para facilitar a iteração
    
    # Plotar os gráficos
    for i, var in enumerate(cat_var):
        var_cat_tim = df_.groupby(['data_ref', var])['index'].count().reset_index()
        sns.lineplot(data=var_cat_tim, x='data_ref', y='index', hue=var, ax=axes[i])
        axes[i].set_title(f"Frequency Over time by {var}")
        axes[i].grid(False)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Remover subplots vazios (se houver)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Ajustar layout geral
    plt.tight_layout()
    plt.show()
    
    
    # #### insight
    # > Here you can see a pattern of behavior in all the variables presented, in which they start to decrease in frequency after the same period. This shows that in certain periods of the year, the frequency changes, which is normal, no anomalies were detected between the different classes, they all behave very similarly.
    
    # In[385]:
    
    
    
    
    # In[386]:
    
    

    
