#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
import lightgbm as lgb 
from  sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import h2o
from h2o.estimators.xgboost import H2OXGBoostEstimator
import plotly.express as px
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output



pd.set_option('display.max_columns', 500)

from sklearn.metrics import plot_confusion_matrix

#df = pd.read_csv('~/Desktop/R/ATP Data.csv')

#df.head()


# In[6]:


def load_atp():
    df = pd.DataFrame
    for i in range(21):
        if i == 0:
            df = pd.read_csv('~/Desktop/R/tennis_atp-master/atp_matches_200'+str(i)+'.csv')
        elif len(str(i))==1:
            df_x = pd.read_csv('~/Desktop/R/tennis_atp-master/atp_matches_200'+str(i)+'.csv')
            df=df.append(df_x, ignore_index=True)
        elif len(str(i))==2:
            df_x = pd.read_csv('~/Desktop/R/tennis_atp-master/atp_matches_20'+str(i)+'.csv')
            df=df.append(df_x, ignore_index=True)
    return df 
df = load_atp()

df.head()


# In[7]:


df['winner_first_serve_percentage'] = df['w_1stIn']/df['w_svpt']

df['winner_first_serve_win_percentage'] = df['w_1stWon']/df['w_1stIn']
df.head()


# In[8]:


df['winner_second_serve_percentage'] = (df['w_svpt']-df['w_1stIn']-df['w_df'])/(df['w_svpt']-df['w_1stIn'])

df['winner_second_serve_win_percentage'] = df['w_2ndWon']/(df['w_svpt']-df['w_1stIn'])

df['winner_second_serves_in'] = (df['w_svpt']-df['w_1stIn']-df['w_df'])

df.head()


# In[9]:


df['loser_first_serve_percentage'] = df['l_1stIn']/df['l_svpt']

df['loser_first_serve_win_percentage'] = df['l_1stWon']/df['l_1stIn']
df.head()


# In[10]:


df['loser_second_serve_percentage'] = (df['l_svpt']-df['l_1stIn']-df['l_df'])/(df['l_svpt']-df['l_1stIn'])

df['loser_second_serve_win_percentage'] = df['l_2ndWon']/(df['l_svpt']-df['l_1stIn'])

df['loser_second_serves_in'] = (df['l_svpt']-df['l_1stIn']-df['l_df'])

df.head()


# In[11]:


len_df = len(df['tourney_id'])-1
df_winners_1 = df.iloc[0:len_df,0:15]
df_winners_2 = df[['score','best_of','round','minutes','w_ace','w_df','w_svpt',
                      'w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved',
                      'w_bpFaced']]
df_winners_3 = df.iloc[0:len_df,-10:-5]
df_winners = pd.concat([df_winners_1,df_winners_2,df_winners_3],axis=1)
df_winners.head()


# In[12]:


df_losers_1 = df.iloc[0:len_df,0:7]
df_losers_2 = df[['loser_id','loser_seed','loser_entry','loser_name','loser_hand',
                 'loser_ht','loser_ioc','loser_age','score','best_of','round','minutes']]
df_losers_3 = df.iloc[0:len_df,36:45]
df_losers_4 = df.iloc[0:len_df,-5:]
df_losers =  pd.concat([df_losers_1,df_losers_2,df_losers_3,df_losers_4],axis=1)
df_losers.head()


# In[13]:


df_losers.columns = df_winners.columns
df = pd.concat([df_winners, df_losers], axis=0, ignore_index=True)
df.head()


# In[14]:


df.tail()


# In[15]:


total_first_serve_percentage = df['w_1stIn'].sum()/df['w_svpt'].sum()
print(total_first_serve_percentage)

total_first_serve_win_percentage = df['w_1stWon'].sum()/df['w_1stIn'].sum()
print(total_first_serve_win_percentage)

total_second_serve_percentage = (df['w_svpt'].sum()-df['w_1stIn'].sum()-df['w_df'].sum())/(df['w_svpt'].sum()-df['w_1stIn'].sum())
print(total_second_serve_percentage)
total_second_serve_win_percentage = (df['w_2ndWon'].sum()/(df['w_svpt'].sum()-df['w_1stIn'].sum()))
print(total_second_serve_win_percentage)


# In[16]:


prob_miss_1st_serve = 1-total_first_serve_percentage
prob_miss_2nd_serve = 1-total_second_serve_percentage
exp_val_2x_1st_serve = (1-(prob_miss_1st_serve**2))*total_first_serve_win_percentage
print(exp_val_2x_1st_serve)


# In[17]:


exp_val_normal = total_first_serve_percentage*total_first_serve_win_percentage+(prob_miss_1st_serve)*total_second_serve_percentage*total_second_serve_win_percentage
print(exp_val_normal)


# In[18]:


#Let's simulate this
import random
def first_serve_2x():
    win = 0
    lose = 0 
    for i in range(1000000):
        x = random.uniform(0,1)
        if x <total_first_serve_percentage:
            y=random.uniform(0,1)
            if y<total_first_serve_win_percentage:
                win+=1
            else:
                lose+=1
        else: 
            x1 = random.uniform(0,1)
            if x1 <total_first_serve_percentage:
                y1=random.uniform(0,1)
                if y1<total_first_serve_win_percentage:
                    win+=1
                else:
                    lose+=1
            else:
                lose+=1
    win_percentage = win/(win+lose)
    return win, lose, win_percentage

attempt = first_serve_2x()

print(attempt)

def normal_serve_2x():
    win = 0
    lose = 0 
    for i in range(1000000):
        x = random.uniform(0,1)
        if x <total_first_serve_percentage:
            y=random.uniform(0,1)
            if y<total_first_serve_win_percentage:
                win+=1
            else:
                lose+=1
        else: 
            x1 = random.uniform(0,1)
            if x1 <total_second_serve_percentage:
                y1=random.uniform(0,1)
                if y1<total_second_serve_win_percentage:
                    win+=1
                else:
                    lose+=1
            else:
                lose+=1
    win_percentage = win/(win+lose)
    return win, lose, win_percentage
            
attempt_normal = normal_serve_2x()
print(attempt_normal)


# In[19]:


df['matches'] = 1
df.head()


# In[20]:


df['exp_val_2x_1st_serve'] = (1-((1-df['winner_first_serve_percentage'])**2))*df['winner_first_serve_win_percentage']
df['exp_val_normal_serve'] = df['winner_first_serve_percentage']*df['winner_first_serve_win_percentage']+(1-df['winner_first_serve_percentage'])*df['winner_second_serve_percentage']*df['winner_second_serve_win_percentage']
df['serve_2x_better'] = np.where(df['exp_val_2x_1st_serve']> df['exp_val_normal_serve'], 1, 0)
df.head()


# In[21]:


print(df['serve_2x_better'].sum()/(len(df['serve_2x_better'])))


# In[22]:


df_players = df[['winner_name',
               'w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms',
               'w_bpSaved','w_bpFaced' , 'matches']]


# In[23]:


df_players = df_players.groupby(by=['winner_name'],sort=False,as_index=False).sum()
df_players.head()


# In[24]:


df_players['winner_first_serve_percentage'] = df_players['w_1stIn']/df_players['w_svpt']

df_players['winner_first_serve_win_percentage'] = df_players['w_1stWon']/df_players['w_1stIn']

df_players['winner_second_serve_percentage'] = (df_players['w_svpt']-df_players['w_1stIn']-df_players['w_df'])/(df_players['w_svpt']-df_players['w_1stIn'])

df_players['winner_second_serve_win_percentage'] = df_players['w_2ndWon']/(df_players['w_svpt']-df_players['w_1stIn'])

df_players['winner_second_serves_in'] = (df_players['w_svpt']-df_players['w_1stIn']-df_players['w_df'])
df_players.head()


# In[25]:


df_players['exp_val_2x_1st_serve'] = (1-((1-df_players['winner_first_serve_percentage'])**2))*df_players['winner_first_serve_win_percentage']
df_players['exp_val_normal_serve'] = df_players['winner_first_serve_percentage']*df_players['winner_first_serve_win_percentage']+(1-df_players['winner_first_serve_percentage'])*df_players['winner_second_serve_percentage']*df_players['winner_second_serve_win_percentage']
df_players['serve_2x_better'] = np.where(df_players['exp_val_2x_1st_serve']> df_players['exp_val_normal_serve'], 1, 0)
df_players.head()


# In[26]:


print(df_players['serve_2x_better'].sum()/(len(df_players['serve_2x_better'])))
print(df_players.shape)

df_players = df_players[df_players.matches>50]
print(df_players.shape)
df_players.head()
print(df_players['serve_2x_better'].sum()/(len(df_players['serve_2x_better'])))


# In[27]:


df['year'] = df['tourney_id'].astype(str).str[0:4]

df.tail()


# In[28]:


player_years = df[['winner_name', 'year',
               'w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms',
               'w_bpSaved','w_bpFaced','matches','winner_ht']]


# In[29]:


player_years = player_years.groupby(by=['winner_name','year','winner_ht'],sort=False,as_index=False).sum()


# In[30]:


player_years['winner_first_serve_percentage'] = player_years['w_1stIn']/player_years['w_svpt']

player_years['winner_first_serve_win_percentage'] = player_years['w_1stWon']/player_years['w_1stIn']

player_years['winner_second_serve_percentage'] = (player_years['w_svpt']-player_years['w_1stIn']-player_years['w_df'])/(player_years['w_svpt']-player_years['w_1stIn'])

player_years['winner_second_serve_win_percentage'] = player_years['w_2ndWon']/(player_years['w_svpt']-player_years['w_1stIn'])

player_years['winner_second_serves_in'] = (player_years['w_svpt']-player_years['w_1stIn']-player_years['w_df'])


# In[31]:


player_years['exp_val_2x_1st_serve'] = (1-((1-player_years['winner_first_serve_percentage'])**2))*player_years['winner_first_serve_win_percentage']
player_years['exp_val_normal_serve'] = player_years['winner_first_serve_percentage']*player_years['winner_first_serve_win_percentage']+(1-player_years['winner_first_serve_percentage'])*player_years['winner_second_serve_percentage']*player_years['winner_second_serve_win_percentage']
player_years['serve_2x_better'] = np.where(player_years['exp_val_2x_1st_serve']> player_years['exp_val_normal_serve'], 1, 0)

player_years['Player_Year'] = player_years['winner_name'] + ", " + player_years['year'].astype(str)
player_years.head()


# In[32]:


nadal = player_years.loc[player_years['winner_name'] == 'Rafael Nadal']


# In[33]:


fig = px.scatter(df_players, x="exp_val_2x_1st_serve",y="exp_val_normal_serve", 
                 hover_name="winner_name",size='matches')
fig


# In[34]:


fig2 = px.line(nadal, x='year',y=['exp_val_2x_1st_serve','exp_val_normal_serve','winner_first_serve_percentage'
                                 ,'winner_first_serve_win_percentage','winner_second_serve_percentage','winner_second_serve_win_percentage'])
fig2


# In[35]:


player_years.sort_values(by=['year'],ignore_index=True)
player_years = player_years.reset_index(drop=True)
player_years = player_years.sort_values(by=['year'],ignore_index=True)



player_years['matches'] = player_years['matches'].astype(float)
player_years = player_years[player_years.matches>10]




player_years['serve_2x_better'] = player_years['serve_2x_better'].astype(float)
player_years = player_years.dropna()

player_years_unpivot = player_years.melt(id_vars=['Player_Year', 'year','winner_name'], var_name='variable', value_name='variable value')


player_years_unpivot['Player_Year'] = player_years_unpivot['Player_Year'].astype(str)
player_years_unpivot['variable'] = player_years_unpivot['variable'].astype(str)
player_years_unpivot['variable value'] = player_years_unpivot['variable value'].astype(float)
player_years_unpivot['year'] = player_years_unpivot['year'].astype(int)

player_years_unpivot


# In[36]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

variable_list = player_years_unpivot['variable'].unique()



app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in variable_list],
                value='exp_val_normal_serve'
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in variable_list],
                value='exp_val_2x_1st_serve'
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Dropdown(
                id='crossfilter-size-dropdown',
                options=[{'label': i, 'value': i} for i in variable_list],
                value='serve_2x_better'
            )
        ], style={'width': '30%', 'display': 'inline-block'})
        
    ], style={
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Roger Federer'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),


])


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-size-dropdown', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,color_name):
    dff = player_years_unpivot

    fig = px.scatter(x=dff[dff['variable'] == xaxis_column_name]['variable value'],
            y=dff[dff['variable'] == yaxis_column_name]['variable value'],
            color=dff[dff['variable'] == color_name]['variable value'],
            hover_name=dff[dff['variable'] == yaxis_column_name]['Player_Year'],
            template='plotly_dark',
            color_continuous_scale = 'Portland'
                    )

    fig.update_traces(customdata=dff[dff['variable'] == yaxis_column_name]['winner_name'])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(dff, title):

    fig = px.scatter(dff, x='year', y='variable value', template='plotly_dark')

    fig.update_traces(mode='lines+markers', marker = {'color' : '#00FE35'})

    fig.update_xaxes(showgrid=False)

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name):
    player_name = hoverData['points'][0]['customdata']
    dff = player_years_unpivot[player_years_unpivot['winner_name'] == player_name]
    dff = dff[dff['variable'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(player_name, xaxis_column_name)
    return create_time_series(dff, title)


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')])
def update_x_timeseries(hoverData, yaxis_column_name):
    dff = player_years_unpivot[player_years_unpivot['winner_name'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['variable'] == yaxis_column_name]
    return create_time_series(dff, yaxis_column_name)


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)


# In[37]:





# In[ ]:





# In[ ]:




