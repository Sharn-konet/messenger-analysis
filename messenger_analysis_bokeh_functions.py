from multiprocessing import Pool
import scrapy
import json
import itertools
import pandas as pd
import numpy as np
from math import pi
from copy import deepcopy
from bokeh.palettes import Category20
from bokeh.models.widgets import Dropdown
from scipy.optimize import curve_fit
from bokeh.models import ColumnDataSource, GroupFilter, CDSView, BoxAnnotation, Panel, Tabs, HoverTool
from bokeh.models.widgets import CheckboxButtonGroup
from bokeh.models.widgets.sliders import DateRangeSlider
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.layouts import column, layout, row, Spacer
from bokeh.plotting import figure, show
from bokeh.transform import cumsum
from datetime import datetime

def parse_html_title(directory):
    with open(directory, 'rb') as data:
        text = data.read()

    data = scrapy.Selector(text=text, type='html')

    title = data.xpath('//title/text()').extract()
    if len(title) == 0:
        return None
    else:
        return title[0]

def parse_html_messages(directory):
    """ Parses an HTML file for messenger data.

        Parameters:
        -----------
        directory - string
            The file path of a particular html file to parse.

        Returns:
        -----------
        message_df - pandas Dataframe
            A dataframe where each row makes up a particular message.
        
        participants - list
            A list of all names included in the chat history as strings.

        reacts - pandas Dataframe
            A dataframe where each row makes up a reaction.
    """

    def extract_message_info(message):
        """ Calls a XPath commands to extract relevant data from the message

            Parameters:
            -----------
            message - scrapy Selector
                A box selected by through scrapy containing all relevant message fields.

            Returns:
            -----------
            message_dict - dictionary
                Dictionary containing the extracted fields.
        """
        message_dict = {'Message': message.xpath('.//div[@class="' + attributes['message'] + '"]/div/div[2]//text()|.//audio/@src|.//a/@href').extract(),
            'Date': message.xpath('.//div[@class="' + attributes['dates'] + '"]/text()').extract()[0],
            'Reacts': message.xpath('.//ul[@class="' + attributes['reacts'] + '"]/li/text()').extract(),
            'Name': message.xpath('.//div[@class="' + attributes['names'] + '"]/text()').extract()[0]}

        message_dict['Reacts'] = [(react[1:], react[0])
                                for react in message_dict['Reacts']]

        message_dict['Date'] = datetime.strptime(
            message_dict['Date'], "%b %d, %Y, %H:%M %p")

        # Should replace with resample to keep messages together and stuff.
        # - Might not work as there are multiple of the same date, so dates dont work as good index
        message_dict['Date'] = message_dict['Date'].replace(hour=0, minute=0)

        return message_dict

    attributes = {
    "names": "_3-96 _2pio _2lek _2lel",
    "dates": "_3-94 _2lem",
    "message": "_3-96 _2let",
    "images": "_2yuc _3-96",
    "reacts": "_tqp",
    "participants": "_2lek",
    "group_name": "_3b0d",
    "plan_name": "_12gz"
    }

    with open(directory, 'rb') as data:
        text = data.read()

    data = scrapy.Selector(text=text, type='html')

    title = data.xpath('//title/text()').extract()[0]

    messages = data.xpath('//div[@class="pam _3-95 _2pi0 _2lej uiBoxWhite noborder"]')[1:]

    messages = [*map(extract_message_info, messages)]

    message_df = pd.DataFrame(messages)

    participants = [*message_df.Name.unique()]

    reacts_list = [*message_df['Reacts']]
    reacts = [react for reacts in reacts_list for react in reacts if len(reacts) > 0]

    reacts = pd.DataFrame(data=reacts, columns=['Name', 'Reacts'])
    reacts = reacts.groupby(["Reacts", "Name"])["Reacts"].count()
    reacts.name = 'Count'
    reacts = reacts.reset_index()

    # Can remove Name column beforehand and then remove the drop = True statement, definitely faster
    # message_df = message_df.set_index('Date').groupby('Name').resample('W', convention='end').apply(lambda x: [*x]).drop(columns="Name").reset_index()
    # message_df.loc[:, 'Message_Count'] = message_df.loc[:, 'Message'].apply(len)

    #data[data['reacts'].apply(len) == max(data['reacts'].apply(len))]
    # - Code which selects all messages which got the maximum number of reacts.
    # Can add an additional ['reacts'] to then see what the reacts are

    return (message_df, reacts, title, participants)

def parse_json_messages(directories):
    """ Parses an HTML file for messenger data.

        Parameters:
        -----------
        directories - list
            A list of directories to concatenate together into one dataset

        Returns:
        -----------
        message_df - pandas Dataframe
            A dataframe where each row makes up a particular message.
        
        reacts - pandas Dataframe
            A dataframe where each row makes up a reaction.

        title - string
            The name of the chat. (Mostly for group chats)
        
        participants - list
            A list of all names included in the chat history as strings.
    """

    def load_json(directory):
        with open(directory.replace("\\", "/"), encoding = 'utf-8') as dataFile:
            message_data = json.load(dataFile)
        
        return message_data

    def rename_message_keys(message):
        
        try:
            message['Message'] = message.pop('content')
            message['Type'] = 'Message'
        except KeyError:
            if 'videos' in message:
                message['Message'] = [video['uri'] for video in message.pop('videos')]
                message['Type'] = 'Video'
            elif 'photos' in message:
                message['Message'] = [photo['uri'] for photo in message.pop('photos')]
                message['Type'] = 'Photo'
            elif 'gifs' in message:
                message['Message'] = [gif['uri'] for gif in message.pop('gifs')]
                message['Type'] = 'Gif'
            elif 'audio_files' in message:
                message['Message'] = [audio_file['uri'] for audio_file in message.pop('audio_files')]
                message['Type'] = 'Audio File'
            elif 'sticker' in message:
                message['Message'] = message.pop('sticker')['uri']
                message['Type'] = 'Sticker'
            elif 'files' in message:
                message['Message'] = [file['uri'] for file in message.pop('files')]
                message['Type'] = 'File'
            else:
                message['Message'] = None
                message['Type'] = None

        message['Name'] = message.pop('sender_name')

        message['Date'] = datetime.fromtimestamp(message.pop('timestamp_ms')/1000)
        message['Date'] = message['Date'].replace(hour=0, minute=0, second = 0, microsecond = 0)

        if 'reactions' in message:
            message['Reacts'] = [(react['actor'], react['reaction'].encode('latin-1').decode('utf-8')) for react in message['reactions']]
        else:
            message['Reacts'] = []

        del message['type']

        return message

    message_data = [*map(load_json, directories)]

    title = message_data[0]['title']

    participants = [*map(lambda name: name['name'], message_data[0]['participants'])]

    message_data = [data['messages'] for data in message_data]

    message_data = [*itertools.chain(*message_data)]

    message_data = [*map(rename_message_keys, message_data)]

    message_df = pd.DataFrame(message_data)

    reacts_list = [*message_df['Reacts']]
    reacts = [react for reacts in reacts_list for react in reacts if len(reacts) > 0]

    reacts = pd.DataFrame(data=reacts, columns=['Name', 'Reacts'])
    reacts = reacts.groupby(["Reacts", "Name"])["Reacts"].count()
    reacts.name = 'Count'
    reacts = reacts.reset_index()

    return (message_df, reacts, title, participants)

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
            legend_label = 'Weekly Total',
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
            'Date': 'datetime'
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
    piechart_figure.title.text_font_size['value'] = '15pt'
    piechart_figure.title.offset = 5

    react_dropdown_menu = [*zip(participants, participants)]

    pie_chart_selection = Dropdown(label="Target Participant", button_type="primary", menu=react_dropdown_menu, sizing_mode = "stretch_both")

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
