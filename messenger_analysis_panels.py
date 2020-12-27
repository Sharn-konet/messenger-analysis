# +==================================================+
# |           messenger_analysis_panels.py           |
# +==================================================+
# Author: Sharn-konet Reitsma

# Description:
# A collection of functions which return the various 
# panels displayed in the Bokeh application. 


import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from functools import partial
from math import pi
from copy import deepcopy
from bokeh.palettes import Category20
from scipy.optimize import curve_fit
from bokeh.models import ColumnDataSource, GroupFilter, CDSView, BoxAnnotation, Panel, HoverTool, Select, DateFormatter, TableColumn, AutocompleteInput, Div, Button
from bokeh.models.widgets import CheckboxButtonGroup
from bokeh.models.widgets.sliders import DateRangeSlider
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models.widgets.tables import DataTable
from bokeh.layouts import column, layout, row, Spacer
from bokeh.plotting import figure
from bokeh.core.properties import Color
from bokeh.transform import cumsum
from datetime import date, timedelta
from itertools import product

def create_title_screen():
    """
    """
    header = Div(text = """
        <h3 align = "center">Welcome to Messenger Analysis!</h3>
        <p><strong></strong></p>
        <p>This is a data visualisation tool for visualising Facebook Messenger data. In order to use it properly you need to download a copy of your own messenger data from Facebook, a tutorial can be found <a href = https://www.facebook.com/help/1701730696756992?helpref=hc_global_nav>here</a>. If you have the JSON version of the download you will get <b>more functionality</b>, but the HTML version is supported as well.<img src="messenger_analysis/statics/test.png" alt="eye-washing" /></p>

        <h4 align = "center">
        Please select which format your downloaded data is in:
        </h4>
        """,
        height_policy = "max",
        sizing_mode = "scale_both",
        align = "end",
        style = {"font-family": 'Verdana'}
    )

    jsonButton = Button(
        align = "center",
        #background = Color((80, 220, 100)),
        button_type = "warning",
        label = "JSON",
        height = 50,
        width = 350,
        sizing_mode = "fixed"
    )
    #jsonButton.on_click(partial(initial_setup, format = "JSON"))

    htmlButton = Button(
        align = "center",
        #background = (80, 220, 100),
        button_type = "success",
        label = "HTML",
        height = 50,
        width = 350,
        sizing_mode = "fixed"   
    )
    #htmlButton.on_click(partial(initial_setup, format = "HTML"))

    introduction_panel = layout(column(children = [header, jsonButton, htmlButton], sizing_mode = "stretch_width"), sizing_mode = "stretch_width")

    return introduction_panel

def create_message_timeseries_fig(message_df, title, participants, colour_palette):
    """ Creates a plot of messages to a chat over time

        Parameters:
        ===========

        message_df - pandas Dataframe
            A dataframe containing data of every message from a chat

        title - string
            A string containing the name of the group chat

        participants - list
            a list of strings of each pariticipant in the messenger chat

        colour_palette - list
            a list of strings which have a participant

        Returns:
        ===========

        message_panel - bokeh Panel
            A layout of graphs for analysis of message data
    """
    
    def aggregate(messages):

        aggregated_messages = messages.groupby(['Date', 'Name']).mean()
        aggregated_messages = aggregated_messages.reset_index(level=1, drop=False)
        aggregated_messages = aggregated_messages.groupby('Name')
        aggregated_messages = aggregated_messages.resample('2W', convention='end').mean().reset_index() # Convert to a weekly aggregation of messages

        return aggregated_messages

    # Find x-axis limits:
    start_date = min(message_df.loc[:, 'Date'])
    end_date = max(message_df.loc[:, 'Date'])

    # Create widget objects:
    main_figure = go.Figure()

    messages = message_df.groupby(['Name', 'Date']).count().reset_index()
    messages = messages.loc[:, messages.columns != 'Reacts']
    
    # Create 0 entries on each day
    existing_entries = {*zip(messages.Name, messages.Date)}
    time_interval = end_date - start_date
    all_days = [start_date + timedelta(days = days) for days in range(time_interval.days + 1)]
    missing_entries = {*product(participants, all_days)} - existing_entries 
    missing_entries = [(participant, day, 0, 0, 0) for participant, day in missing_entries]
    missing_entries = pd.DataFrame(data = missing_entries, columns = messages.columns)

    messages = messages.append(missing_entries)

    agg_messages = aggregate(messages)
    agg_messages.loc[agg_messages.loc[:,'Message'] == 0, 'Message'] = None

    # Plot a line for each person onto both figures:
    for index, name in enumerate(participants):
        view = agg_messages[agg_messages['Name'] == name]

        main_figure.add_trace(go.Scatter(x=view['Date'], y=view['Message'],
            name = name,
            line_shape = 'spline',
            opacity = 0.55,
            line = dict(color=colour_palette[index]),
            mode = 'lines+markers'
        ))

    main_figure.update_layout(
        xaxis_title = "Time",
        yaxis_title = "Total Messages",
        legend_title = "Participants",
        transition_duration = 800,
        margin = {
            't': 30
        }
    )

    graph = dcc.Graph(id = 'timeline', figure = main_figure,
                      config = {
                          "autosizable": True,
                          #"fillFrame": True,
                          "showTips": True,
                          "showAxisDragHandles": True,
                          "showAxisRangeEntryBoxes": True
                      })

    return html.Div(graph, style = {"height": "100%"})

def create_react_breakdown_panel(reacts, title, participants, colour_palette):

    # def update_pie_chart(attr, old, new):
    #     df = deepcopy(reacts_individual)
    #     df = df[df['Name'] == new]
    #     reacts_indiv_CDS.data = df
    #     piechart_figure.title.text = "Distribution of Reactions for " + str(new)

    unique_reacts = reacts['Reacts'].unique()

    react_color = dict(zip(unique_reacts, Category20[20][0:len(unique_reacts)]))

    reacts_individual = reacts.groupby(["Name", "Reacts"]).sum().reset_index()

    sums = {name: (reacts_individual[reacts_individual['Name'] == name].loc[:, 'Count'].sum()) for name in participants}

    reacts_individual = reacts_individual.apply(lambda x: pd.Series([*x, (x[2]/sums[x[0]])*2*pi, react_color[x[1]]], index = ['Name', 'Reacts', 'Count', 'Angle', 'Color']), axis = 1)

    reacts = reacts.pivot(index='Reacts', columns='Name', values='Count').fillna(0)

    sums = reacts.sum(axis=1)

    for i in reacts.index:
        reacts.loc[i, :] = reacts.loc[i, :].apply(lambda x: (x/sums[i]))

    react_pie_fig = go.Figure(
        data=[go.Pie(
            labels = list({*reacts_individual['Reacts']}),
            values = reacts_individual[reacts_individual['Name'] == participants[0]]['Count'],
            marker = {"colors": colour_palette[:len(participants)]}
        )]
    )

    react_pie_fig.update_layout(
        title = "Breakdown of Reacts from {}".format(participants[0]),
        margin = {
            't': 80,
            'l': 15
        }
    )

    react_bar_fig = go.Figure(
        [go.Bar(name = individual, 
                x = reacts[individual].index.values, 
                y = reacts[individual].values,
                marker = {"color": colour_palette[colour]}) 
        for colour, individual in enumerate(participants)]
    )

    react_tooltip = [
        ("Name", "$name"),
        ("React", "@Reacts"),
        ("Percentage", "@$name{ 0.0%}")
    ]

    react_bar_fig.update_layout(
        margin = {'t': 30,
                  'r': 15,
                  'l': 50},
        barmode = 'stack'
    )

    graphs = [
        dcc.Graph(id = 'react-graph', figure = react_bar_fig, style = {'width': "60%", "height": "75%", 'flex': '1'}),
        dcc.Graph(id = 'react-bar', figure = react_pie_fig, style = {'width': "30%", "height": "75%"})
    ]

    return html.Div(graphs, style = {'display':'flex'})

def create_message_log_panel(message_df, participants):

    # Create DataTable Widget:
    message_data = message_df[message_df['Type']=='Message']
    del message_data['Type']
    del message_data['Details']

    table = html.Div(
        children = [dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": column, 'id': column} for column in message_data.columns
        ],
        tooltip_data=[
        {
            column: {'value': str(value), 'type': 'markdown'}
            for column, value in row.items() if column == 'Message'
        } for row in message_data.to_dict('records')
        ],
        tooltip_duration = None,
        data=message_data.to_dict('records'),
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_deletable=False,
        style_as_list_view = True,
        style_header = {
            'fontWeight': 'bold'
        },
        style_cell = {
            'padding' : "0.5% 0.75% 0.5% 0.75%",
            'textOverflow': 'ellipsis',
            'maxWidth': '180px'
        },
        page_action="native",
        page_current= 0,
        page_size= 10,
    )],
    style = {"margin": "2%", "border-style": "groove", "border-width": "2px", "border-color": "lightgrey"}
    )

    return table

def create_individual_statistics_panel(message_df, title, participants, colour_palette):
    """ Create a panel which summarises the individual statistics of any user within the selected group.
    """

    all_messages = message_df.loc[message_df['Type'] == 'Message', 'Message'].reset_index()['Message']

    lengths = [*map(len, all_messages)]
    print("The maximum length message you've ever sent is: {}".format(max(lengths)))
    print("\nThe message was: \n {}".format(all_messages.loc[lengths.index(max(lengths))]))

    lengths.sort()

    # From this it seems like 70 characters would be a good amount of characters to use
    plt.hist(lengths[:int(len(lengths)*0.95)], bins = len({*lengths[:int(len(lengths)*0.95)]}))
    plt.show()