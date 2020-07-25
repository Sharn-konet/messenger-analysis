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

from glob import glob

from bokeh.io import curdoc
from bokeh.plotting import show
from bokeh.palettes import Category20
from bokeh.models import Tabs, AutocompleteInput, Div
from bokeh.layouts import column, layout, row, Spacer

from messenger_analysis_panels import message_timeseries_panel, react_breakdown_panel, individual_statistics_panel
from messenger_analysis_data_functions import parse_html_title, parse_html_messages, parse_json_messages, get_chat_titles, update_data

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
#html_names = {name: parse_html_title(directory) for name, directory in html_directories.items()}
chat_titles = {get_chat_titles(json_directories, key): key for key in json_directories}

directory = html_directories['THELOVECHAT_BT-aNw8Nzg']

json_directory = json_directories['THELOVECHAT_BT-aNw8Nzg']

#'MaxPitto_V4hEVp1UCQ'

#(message_df, reacts, title, participants) = parse_html_messages(directory)

(message_df, reacts, title, participants) = parse_json_messages(json_directory)

# -------------------------------------------------------------------------
# Plot Message Timeseries:
# -------------------------------------------------------------------------

# Create a color palette to use in plotting:
""" Might raise an error when number of people in the group chat is > 20"""
colour_palette = Category20[20][0:len(participants)]

message_panel = message_timeseries_panel(message_df, title, participants, colour_palette)

# --------------------------------------------------------------------------+
# Plot Reaction Panel:
# --------------------------------------------------------------------------+

reacts_panel = react_breakdown_panel(reacts, title, participants, colour_palette)

# --------------------------------------------------------------------------+
# Create Panel to Summarise Individual Statistics:
# --------------------------------------------------------------------------+

individual_statistics_panel = individual_statistics_panel(message_df, title, participants, colour_palette)

# --------------------------------------------------------------------------+
# Compile Bokeh Application:
# --------------------------------------------------------------------------+

tabs = Tabs(tabs=[message_panel, reacts_panel, individual_statistics_panel])

directory_search = AutocompleteInput(completions = list(chat_titles.keys()), width = 400, height = 30, sizing_mode = "fixed", align = 'end')
directory_search.on_change("value", update_data)

search_text = Div(
    text = "<i>Search Chats:</i>",
    height_policy = "max",
    sizing_mode = "scale_both",
    align = "end",
    style = {"font-family": 'Verdana', "font-size": "17px"}
)

# A title which could be included in the top left of the document
title = Div(
    text = "<b>Messenger Analysis</b>",
    height_policy = "max",
    sizing_mode = "fixed",
    align = "start",
    style = {"font-family": 'Verdana', "font-size": "16px"}
)

layout = column(row(search_text,directory_search, Spacer(
        width=35, height=40, sizing_mode="fixed"), align = "end"), tabs, sizing_mode = "scale_width")

show(layout)

curdoc().title = "Messenger Analysis"
curdoc().add_root(layout)
