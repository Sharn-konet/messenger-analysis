# +=================================================+
# |           messenger_analysis_bokeh.py           |
# +=================================================+
# Author: Sharn-konet Reitsma

# Description:
# This script is an alternative implementation of my original messenger analysis script
# and it's corresponding functions. It uses XPath to parse the downloaded HTML files,
# which is much faster than its BeautifulSoup equivalent.

# In addition to this I've also implemented all plots into Bokeh, such that I can introduce
# more interactivity with the plots, giving people a bit more power to do their own analyses.

# --------------------------------------------------------------------------------------------
# SCRIPT START
# --------------------------------------------------------------------------------------------

from glob import glob
import scrapy
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from math import pi
from scipy.optimize import curve_fit
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, GroupFilter, CDSView, BoxAnnotation, Panel, Tabs, HoverTool
from bokeh.models.widgets import CheckboxButtonGroup
from bokeh.palettes import Spectral11, Category20, viridis
from bokeh.models.glyphs import MultiLine
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row, Spacer
from bokeh.models.widgets.sliders import DateRangeSlider
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.transform import cumsum
from bokeh.core.properties import field
from messenger_analysis_bokeh_functions import parse_html_messages, parse_json_messages

# -------------------------------------------------------------------------
# Parsing Messenger Files:
# -------------------------------------------------------------------------

html_directories = glob("**/messages/inbox/*/*.html", recursive=True)
html_directories = {directory.split(
    "\\")[-2]: directory for directory in html_directories}
message_names = {key: key.split("_")[0] for key in html_directories}

json_directories = glob("**/messages/inbox/*/*.json", recursive = True)
unique_chatnames = {*map(lambda dir: dir.split("\\")[-2], json_directories)}
json_directories = {name: [directory for directory in json_directories if name in directory] for name in unique_chatnames}

directory = html_directories['THELOVECHAT_BT-aNw8Nzg']

json_directory = json_directories['THELOVECHAT_BT-aNw8Nzg']

(message_df, reacts, title, participants) = parse_html_messages(directory)

#(message_df, reacts, title, participants) = parse_json_messages(json_directory)

# -------------------------------------------------------------------------
# Plot Data:
# -------------------------------------------------------------------------

# Find x-axis limits:
start_date = min(message_df.loc[:, 'Date'])
end_date = max(message_df.loc[:, 'Date'])

# Create widget objects:
name_buttons = CheckboxButtonGroup(labels=participants, active=[
                                   i for i in range(len(participants))])
date_slider = DateRangeSlider(
    end=end_date, start=start_date, value=(start_date, end_date), step=1)

# Create a color palette to use in plotting:
# mypalette = viridis(len(participants))
mypalette = Category20[20][0:len(participants)]

# Create figures to be included:
p = figure(plot_width=800, plot_height=250,
           x_axis_type="datetime", toolbar_location=None)
p.toolbar.logo = None
p.x_range.start = start_date
p.x_range.end = end_date
p.toolbar.active_drag = None
p.toolbar.active_scroll = None

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

p.add_tools(messages_tooltip)

p2 = figure(plot_height=80, plot_width=800, x_axis_type='datetime', toolbar_location=None,
            x_range=(start_date, end_date))
p2.yaxis.major_label_text_color = None
p2.yaxis.major_tick_line_color = None
p2.yaxis.minor_tick_line_color = None
p2.grid.grid_line_color = None
p2.toolbar.active_drag = None
p2.toolbar.active_scroll = None

box = BoxAnnotation(fill_alpha=0.5, line_alpha=0.5,
                    level='underlay', left=start_date, right=end_date)

messages = message_df.groupby(['Name', 'Date']).count().reset_index()
messages = messages.loc[:, messages.columns != 'Reacts']

source = ColumnDataSource(data=messages)

# Plot a line for each person onto both figures:
for i in range(len(participants)):
    view = CDSView(source=source,
                   filters=[GroupFilter(column_name='Name', group=participants[i])])
    p.line(
        x='Date',
        y='Message',
        source=source,
        view=view,
        alpha=0.45,
        muted_color=mypalette[i],
        muted_alpha=0.2,
        legend_label=participants[i],
        color=mypalette[i]
    )

    p.circle(
        x='Date',
        y='Message',
        alpha=0.55,
        source=source,
        view=view,
        muted_color=mypalette[i],
        muted_alpha=0.2,
        legend_label=participants[i],
        color=mypalette[i]
    )

    p2.line(
        x='Date',
        y='Message',
        source=source,
        view=view,
        color=mypalette[i]
    )

total_messages = messages.groupby('Date').mean()
total_messages = total_messages.resample('W', convention='end').sum().reset_index() # Convert to a weekly aggregation of messages

# See if your chat is dying:
def curve(x, a, b, c):
    return a * np.exp(-b * x) + c

x_data = total_messages.Date.array.asi8/ 1e18
y_data = total_messages['Message'].values

popt, pcov = curve_fit(curve, x_data, y_data, maxfev = 20000)

x_data_step = x_data[1] - x_data[0]

x_data = list(x_data)

for _ in range(51):
    x_data.append(x_data[-1] + x_data_step)

x_data = np.array(x_data)

y_prediction = curve(x_data, *popt)

total_messages['Prediction'] = y_prediction[:len(total_messages)]

total_messages_cds = ColumnDataSource(data=total_messages)

p.line(
    x='Date',
    y='Message',
    source=total_messages_cds,
    alpha=0.45,
    muted_alpha=0.2,
    legend_label = 'Total Messages',
    color = 'black'
    )

p.circle(
    x='Date',
    y='Message',
    source=total_messages_cds,
    alpha=0.45,
    muted_alpha=0.2,
    legend_label = 'Total Messages',
    color = 'black'
    )

p.line(
    x='Date',
    y='Prediction',
    source=total_messages_cds,
    alpha=0.45,
    muted_alpha=0.2,
    legend_label = 'Prediction',
    line_dash = 'dashed',
    color = 'red'
)

p.xaxis.axis_label = 'Time'
p.yaxis.axis_label = 'Total Messages'
p.title.text = title
p.legend.location = "top_right"
p.legend.click_policy = "mute"

p2.add_layout(box)

# --------------------------------------------------------------------------+
# Creating Real Python callbacks for interaction between plots and widgets |
# --------------------------------------------------------------------------+


def subset_data(initial=False):
    df = message_df
    selected_names = [name_buttons.labels[i] for i in name_buttons.active]
    df = df[df['Name'].isin(selected_names)]
    df = df.groupby(by=['Name', 'Date']).count().reset_index()
    return df


def update_graph(active_labels):
    print(active_labels)
    df = subset_data()
    source.data = dict(
        Date=df['Date'],
        Message=df['Message'],
        Name=df["Name"],
    )


def update_range(attr, old, new):
    #df = subset_data()
    start = datetime.fromtimestamp(new[0]/1e3)
    end = datetime.fromtimestamp(new[1]/1e3)
    #df = df[df['Date'] >= start]
    #df = df[df['Date'] <= end]
    p.x_range.start = start
    p.x_range.end = end

    box.left = start
    box.right = end

    # source.data = dict(
    #     Dates=df['Date'],
    #     Message=df['Message'],
    #     Names=df["Names"],
    # )


# Assign callbacks to appropriate widget interactions
name_buttons.on_click(update_graph)
date_slider.on_change('value', update_range)

date_slider_layout = row(Spacer(
    width=46, height=50, sizing_mode="fixed"), date_slider, sizing_mode="scale_width")

plots = column(p, date_slider_layout, p2, sizing_mode="scale_width")

# Create the layout of the Bokeh application
message_timeseries = layout([
    [name_buttons],
    [plots]
], sizing_mode="scale_width")

message_panel = Panel(child=message_timeseries, title='Message Data')

# --------------------------------------------------------------------------+
# Plotting reactions
# --------------------------------------------------------------------------+
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

p3 = figure(plot_width=800, plot_height=250, x_range=unique_reacts,
            y_range=[0, 1], toolbar_location=None, tooltips=react_tooltip)
p3.xaxis.major_label_text_font_size = "25pt"
p3.toolbar.active_drag = None
p3.toolbar.active_scroll = None

p4 = figure(plot_width=400, plot_height=400, x_range=(-0.5, 1),
            toolbar_location=None, tools="hover", tooltips="@React: @$name{ 0.0%}")

for i in range(len(participants)):
    view = CDSView(source=reacts_indiv_CDS,
                   filters=[GroupFilter(column_name='Name', group=participants[i])])

    p4.wedge(x=0, y=1, radius=0.4,
             source=reacts_indiv_CDS, view=view,
             start_angle=cumsum('Angle', include_zero=True), end_angle=cumsum('Angle'),
             line_color="white", fill_color='Color', legend_field= 'Reacts')

p4.xgrid.grid_line_color = None
p4.ygrid.grid_line_color = None
p4.toolbar.active_drag = None
p4.toolbar.active_scroll = None

# configure so that Bokeh chooses what (if any) scroll tool is active

p3.vbar_stack(
    participants,
    x="Reacts",
    width=0.6,
    source=reacts_source,
    legend_label=participants,
    color=mypalette,
    fill_alpha=0.5
)

p3.yaxis.formatter = NumeralTickFormatter(format="0%")
legend = p3.legend[0]
legend.orientation = 'horizontal'
legend.location = 'center_right'
legend.spacing = 18
p3.add_layout(legend, 'above')


reacts_panel = layout([
    [p4, p3]
], sizing_mode="scale_width")

reacts_panel = Panel(child=reacts_panel, title='Reacts Data')

tabs = Tabs(tabs=[message_panel, reacts_panel])

show(tabs)

curdoc().add_root(tabs)

show(message_timeseries)
