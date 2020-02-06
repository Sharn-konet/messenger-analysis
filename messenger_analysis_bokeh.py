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
from bokeh.plotting import show
from bokeh.models import ColumnDataSource, GroupFilter, CDSView, BoxAnnotation, Panel, Tabs, HoverTool
from bokeh.models.annotations import Title
from bokeh.palettes import Spectral11, Category20, viridis
from bokeh.models.glyphs import MultiLine
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row, Spacer
from bokeh.core.properties import field
from messenger_analysis_bokeh_functions import parse_html_title, parse_html_messages, parse_json_messages, create_message_timeseries_panel, create_react_breakdown_panel

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

directory = html_directories['THELOVECHAT_BT-aNw8Nzg']

json_directory = json_directories['THELOVECHAT_BT-aNw8Nzg']

#(message_df, reacts, title, participants) = parse_html_messages(directory)

(message_df, reacts, title, participants) = parse_json_messages(json_directory)

# -------------------------------------------------------------------------
# Plot Message Timeseries:
# -------------------------------------------------------------------------

# Create a color palette to use in plotting:
""" Might raise an error when number of people in the group chat is > 20"""
colour_palette = Category20[20][0:len(participants)]

message_panel = create_message_timeseries_panel(message_df, title, participants, colour_palette)

# --------------------------------------------------------------------------+
# Plot Reaction Panel:
# --------------------------------------------------------------------------+

reacts_panel = create_react_breakdown_panel(reacts, title, participants, colour_palette)

# --------------------------------------------------------------------------+
# Compile Bokeh Application:
# --------------------------------------------------------------------------+

tabs = Tabs(tabs=[message_panel, reacts_panel])

show(tabs)

curdoc().add_root(tabs)
