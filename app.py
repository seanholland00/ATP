#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output



player_years_unpivot = pd.read_csv("player_years_unpivot.csv")



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

variable_list = player_years_unpivot['variable'].unique()



app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Markdown('''
                # ATP Serve Data 2000-2020
                ### An Analysis of the Two-First-Serve Strategy
                ###### Select an x-variable, y-variable, and color-variable
            ''')
        ]),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in variable_list],
                value='Expected Value: First-Second Serve'
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in variable_list],
                value='Expected Value: Two-First-Serves'
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Dropdown(
                id='crossfilter-size-dropdown',
                options=[{'label': i, 'value': i} for i in variable_list],
                value='Two-First Serves has Higher Exp Val'
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

    fig.update_traces(customdata=dff[dff['variable'] == yaxis_column_name]['Player Name'])

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
    dff = player_years_unpivot[player_years_unpivot['Player Name'] == player_name]
    dff = dff[dff['variable'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(player_name, xaxis_column_name)
    return create_time_series(dff, title)


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')])
def update_x_timeseries(hoverData, yaxis_column_name):
    dff = player_years_unpivot[player_years_unpivot['Player Name'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['variable'] == yaxis_column_name]
    return create_time_series(dff, yaxis_column_name)


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)