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
from datetime import datetime

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

def create_message_timeseries_panel(message_df, title, participants, colour_palette):
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
    
    def plot_summary_data(messages):

        # See if your chat is dying:
        def curve(x, a, b, c):
            return a * np.exp(-b * x) + c

        total_messages = messages.groupby('Date').mean()
        total_messages = total_messages.resample('W', convention='end').sum().reset_index() # Convert to a weekly aggregation of messages

        x_data = total_messages.Date.array.asi8/ 1e18
        y_data = total_messages['Message'].values

        popt, _ = curve_fit(curve, x_data, y_data, maxfev = 100000)

        x_data_step = x_data[1] - x_data[0]

        x_data = list(x_data)

        for _ in range(51):
            x_data.append(x_data[-1] + x_data_step)

        x_data = np.array(x_data)

        y_prediction = curve(x_data, *popt)

        total_messages['Prediction'] = y_prediction[:len(total_messages)]

        total_messages_cds = ColumnDataSource(data=total_messages)

        main_figure.line(
            x='Date',
            y='Message',
            source=total_messages_cds,
            alpha=0.45,
            muted_alpha=0.2,
            legend_label = 'Weekly Total',
            color = 'black'
            )

        main_figure.circle(
            x='Date',
            y='Message',
            source=total_messages_cds,
            alpha=0.45,
            muted_alpha=0.2,
            legend_label = 'Weekly Average',
            color = 'black'
            )

        main_figure.line(
            x='Date',
            y='Prediction',
            source=total_messages_cds,
            alpha=0.45,
            muted_alpha=0.2,
            legend_label = 'Trend',
            line_dash = 'dashed',
            color = 'red'
        )

    # Find x-axis limits:
    start_date = min(message_df.loc[:, 'Date'])
    end_date = max(message_df.loc[:, 'Date'])

    # Create widget objects:
    name_buttons = CheckboxButtonGroup(labels=participants, active=[
                                    i for i in range(len(participants))])
    date_slider = DateRangeSlider(
        end=end_date, start=start_date, value=(start_date, end_date), step=1)

    # Create figures to be included:
    main_figure = figure(plot_width=800, plot_height=250,
            x_axis_type="datetime", toolbar_location=None)
    main_figure.toolbar.logo = None
    main_figure.x_range.start = start_date
    main_figure.x_range.end = end_date
    main_figure.toolbar.active_drag = None
    main_figure.toolbar.active_scroll = None

    messages_tooltip = HoverTool(
        tooltips=[
            ('Name', '@Name'),
            ('Message Count', '@Message'),
            ('Date', '@Date{%A, %e %B %Y}')
        ],
        formatters={
            '@Date': "datetime"
        }
    )

    main_figure.add_tools(messages_tooltip)

    overview_figure = figure(plot_height=80, plot_width=800, x_axis_type='datetime', toolbar_location=None,
                x_range=(start_date, end_date))
    overview_figure.yaxis.major_label_text_color = None
    overview_figure.yaxis.major_tick_line_color = None
    overview_figure.yaxis.minor_tick_line_color = None
    overview_figure.grid.grid_line_color = None
    overview_figure.toolbar.active_drag = None
    overview_figure.toolbar.active_scroll = None

    box = BoxAnnotation(fill_alpha=0.5, line_alpha=0.5,
                        level='underlay', left=start_date, right=end_date)

    messages = message_df.groupby(['Name', 'Date']).count().reset_index()
    messages = messages.loc[:, messages.columns != 'Reacts']

    source = ColumnDataSource(data=messages)

    # Plot a line for each person onto both figures:
    for index, name in enumerate(participants):
        view = CDSView(source=source,
                    filters=[GroupFilter(column_name='Name', group=name)])

        main_figure.line(
            x='Date',
            y='Message',
            source=source,
            view=view,
            alpha=0.45,
            muted_color=colour_palette[index],
            muted_alpha=0.2,
            legend_label=name,
            color=colour_palette[index]
        )

        main_figure.circle(
            x='Date',
            y='Message',
            alpha=0.55,
            source=source,
            view=view,
            muted_color=colour_palette[index],
            muted_alpha=0.2,
            legend_label= name,
            color=colour_palette[index]
        )

        overview_figure.line(
            x='Date',
            y='Message',
            source=source,
            view=view,
            color=colour_palette[index]
        )

    if len(participants) > 2:
        plot_summary_data(messages)

    main_figure.xaxis.axis_label = 'Time'
    main_figure.yaxis.axis_label = 'Total Messages'
    main_figure.title.text = title
    main_figure.legend.location = "top_right"
    main_figure.legend.click_policy = "mute"
    main_figure.legend.orientation = 'horizontal'
    main_figure.legend.spacing = 7

    overview_figure.add_layout(box)

    # --------------------------------------------------------------------------+
    # Creating Real Python callbacks for interaction between plots and widgets |
    # --------------------------------------------------------------------------+

    def update_graph(active_labels):
        df = message_df
        selected_names = [name_buttons.labels[i] for i in name_buttons.active]
        df = df[df['Name'].isin(selected_names)]
        df = df.groupby(by=['Name', 'Date']).count().reset_index()
        source.data = dict(
            Date=df['Date'],
            Message=df['Message'],
            Name=df["Name"],
        )


    def update_range(attr, old, new):
        start = datetime.fromtimestamp(new[0]/1e3)
        end = datetime.fromtimestamp(new[1]/1e3)
        main_figure.x_range.start = start
        main_figure.x_range.end = end

        box.left = start
        box.right = end

    # Assign callbacks to appropriate widget interactions
    name_buttons.on_click(update_graph)
    date_slider.on_change('value', update_range)

    date_slider_layout = row(Spacer(
        width=46, height=50, sizing_mode="fixed"), date_slider, sizing_mode="scale_width")

    plots = column(main_figure, date_slider_layout, overview_figure, sizing_mode="scale_width")

    # Create the layout of the Bokeh application
    message_timeseries = layout([
        [name_buttons],
        [plots]
    ], sizing_mode="scale_width")

    message_timeseries.margin = (10, 35, 60, 20)

    message_panel = Panel(child=message_timeseries, title='Message Data')

    return message_panel


def create_react_breakdown_panel(reacts, title, participants, colour_palette):

    def update_pie_chart(attr, old, new):
        df = deepcopy(reacts_individual)
        df = df[df['Name'] == new]
        reacts_indiv_CDS.data = df
        piechart_figure.title.text = "Distribution of Reactions for " + str(new)

    unique_reacts = reacts['Reacts'].unique()

    react_color = dict(zip(unique_reacts, Category20[20][0:len(unique_reacts)]))

    reacts_individual = reacts.groupby(["Name", "Reacts"]).sum().reset_index()

    sums = {name: (reacts_individual[reacts_individual['Name'] == name].loc[:, 'Count'].sum()) for name in participants}

    reacts_individual = reacts_individual.apply(lambda x: pd.Series([*x, (x[2]/sums[x[0]])*2*pi, react_color[x[1]]], index = ['Name', 'Reacts', 'Count', 'Angle', 'Color']), axis = 1)

    reacts_indiv_CDS = ColumnDataSource(reacts_individual)

    reacts = reacts.pivot(index='Reacts', columns='Name', values='Count').fillna(0)

    sums = reacts.sum(axis=1)

    for i in reacts.index:
        reacts.loc[i, :] = reacts.loc[i, :].apply(lambda x: (x/sums[i]))

    reacts_source = ColumnDataSource(reacts)

    react_tooltip = [
        ("Name", "$name"),
        ("React", "@Reacts"),
        ("Percentage", "@$name{ 0.0%}")
    ]

    bargraph_figure = figure(plot_width=700, plot_height=300, x_range=unique_reacts,
                y_range=[0, 1], toolbar_location=None, tooltips=react_tooltip, sizing_mode = "scale_both")

    bargraph_figure.xaxis.major_label_text_font_size = "25pt"
    bargraph_figure.toolbar.active_drag = None
    bargraph_figure.toolbar.active_scroll = None

    bargraph_figure.vbar_stack(
        participants,
        x="Reacts",
        width=0.6,
        source=reacts_source,
        legend_label=participants,
        color=colour_palette,
        fill_alpha=0.5
    )

    bargraph_figure.yaxis.formatter = NumeralTickFormatter(format="0%")
    legend = bargraph_figure.legend[0]
    legend.orientation = 'horizontal'
    legend.location = 'center_right'
    legend.spacing = 7
    bargraph_figure.add_layout(legend, 'above')
    bargraph_figure.legend.label_text_font_size = "8pt"

    piechart_tooltip = [
        ("React", "@Reacts"),
        ("Number of Reacts", "@Count")
    ]

    piechart_figure = figure(plot_width=423, plot_height=423, x_range=(-0.5, 1),
                toolbar_location=None, tools="hover", tooltips=piechart_tooltip, sizing_mode = "fixed")

    piechart_figure.title.align = 'center'
    piechart_figure.title.border_line_dash = 'solid'
    piechart_figure.title.text_font_size = '15px'
    piechart_figure.title.offset = 5

    pie_chart_selection = Select(value = participants[0], title = "Target Participant", options = participants, sizing_mode = "stretch_both")

    pie_chart_selection.on_change("value", update_pie_chart)

    for name in participants:
        view = CDSView(source=reacts_indiv_CDS,
                    filters=[GroupFilter(column_name='Name', group=name)])

        piechart_figure.wedge(x=0.1, y=1, radius=0.5,
                source=reacts_indiv_CDS, view=view,
                start_angle=cumsum('Angle', include_zero=True), end_angle=cumsum('Angle'),
                line_color="white", fill_color='Color', legend_field= 'Reacts')

    react_page_spacer = Spacer(height = 200, height_policy = "fit", margin = (50,50,50,50))

    update_pie_chart('value', participants[0], participants[0])

    piechart_figure.xgrid.grid_line_color = None
    piechart_figure.ygrid.grid_line_color = None
    piechart_figure.toolbar.active_drag = None
    piechart_figure.toolbar.active_scroll = None
    piechart_figure.axis.axis_label = None
    piechart_figure.axis.visible = False
    piechart_figure.grid.grid_line_color = None

    piechart_figure.legend.label_text_font_size = '15pt'
    piechart_figure.legend.spacing = 19

    react_indiv_column = column(piechart_figure, pie_chart_selection, sizing_mode = "fixed")

    reacts_panel = layout([
        [bargraph_figure, react_indiv_column],
        [react_page_spacer]
    ], sizing_mode="scale_both")

    reacts_panel.margin = (10, 35, 60, 20)

    reacts_panel = Panel(child=reacts_panel, title='Reacts Data')

    return reacts_panel

def create_message_log_panel(message_df, title, participants, colour_palette):

    def filter_for_user(attr, old, new):
        df = deepcopy(message_df)
        df = df[df['Type']=='Message']
        if new in participants:
            df = df[df['Name'] == new]
            message_CDS.data = df
        elif new == 'all':
            message_CDS.data = df

    # Old version, probably not so relevant anymore
    type_counts = message_df.groupby(['Name', 'Type']).count().reset_index()

    type_counts = ColumnDataSource(type_counts)

    type_bar_graph = figure(x_range= [*message_df['Type'].unique()] , plot_height=350, title="Distribution of Types",
           toolbar_location=None, tools="")

    type_bar_graph.vbar(source = type_counts, x='Type', top='Message', width=0.9)

    type_bar_graph.xgrid.grid_line_color = None
    type_bar_graph.y_range.start = 0

    # Create DataTable Widget:

    columns = [
        TableColumn(field = "Message", title = "Message"),
        TableColumn(field = "Name", title = "Name", width = 10),
        TableColumn(field = "Date", title = "Date", formatter = DateFormatter(format = "%d/%m/%Y"), width = 10)
    ]

    message_CDS = ColumnDataSource(message_df[message_df['Type']=='Message'])

    data_table = DataTable(source = message_CDS, columns = columns, fit_columns = True, width = 700, height = 350)

    directory_search = AutocompleteInput(completions = participants, width = 400, height = 30, sizing_mode = "fixed", align = 'start')
    directory_search.on_change("value", filter_for_user)

    filter_text = Div(
        text = "Filter for a Particular User:",
        height_policy = "max",
        sizing_mode = "scale_both",
        align = "end",
        style = {"font-family": 'Verdana', "font-size": "17px"}
    )
    
    filter_input = row(filter_text, directory_search)

    message_log_panel = layout([
        column(filter_input, data_table, sizing_mode = "scale_both")
    ], sizing_mode = "scale_both")

    message_log_panel = Panel(child=message_log_panel, title='Message Log')

    return message_log_panel

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