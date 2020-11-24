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
from bokeh.document import Document

from messenger_analysis_panels import create_message_timeseries_panel, create_react_breakdown_panel, create_message_log_panel, create_title_screen
from messenger_analysis_data_functions import parse_html_title, parse_html_messages, parse_json_messages, get_chat_titles, update_data, create_document

# -------------------------------------------------------------------------
# Parsing Messenger Files:
# -------------------------------------------------------------------------

show(create_title_screen())

introduction_panel = create_title_screen()

html_directories = glob("**/messages/inbox/*/*.html", recursive = True)
html_directories = {directory.split("\\")[-2]: directory for directory in html_directories}
message_names = {key: key.split("_")[0] for key in html_directories}

json_directories = glob("**/messages/inbox/*/*.json", recursive = True)
unique_chatnames = {*map(lambda dir: dir.split("\\")[-2], json_directories)}
json_directories = {name: [directory for directory in json_directories if name in directory] for name in unique_chatnames}
#html_names = {name: parse_html_title(directory) for name, directory in html_directories.items()}
chat_titles = {get_chat_titles(json_directories, key): key for key in json_directories}

directory = html_directories['HelenChambers_iBdYJMhaNw']

json_directory = json_directories['HelenChambers_iBdYJMhaNw']

#'MaxPitto_V4hEVp1UCQ'

#(message_df, reacts, title, participants) = parse_html_messages(directory)

layout = create_document(json_directory, chat_titles)

show(layout)

curdoc().title = "Messenger Analysis"

curdoc().add_root(layout)
