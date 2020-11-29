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

import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output

from bokeh.io import curdoc
from bokeh.plotting import show
from bokeh.palettes import Category20
from bokeh.models import Tabs, AutocompleteInput, Div
from bokeh.layouts import column, layout, row, Spacer
from bokeh.document import Document
from dash_html_components.H1 import H1

from messenger_analysis_panels import create_message_timeseries_fig, create_react_breakdown_panel, create_message_log_panel, create_title_screen
from messenger_analysis_data_functions import parse_html_title, parse_html_messages, parse_json_messages, get_chat_titles, update_data, create_document

# -------------------------------------------------------------------------
# Parsing Messenger Files:
# -------------------------------------------------------------------------

app = dash.Dash(__name__, 
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
                title = "Messenger Analysis")



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

json_directory = json_directories['THELOVECHAT_BT-aNw8Nzg']

#'MaxPitto_V4hEVp1UCQ'
#'THELOVECHAT_BT-aNw8Nzg'

#(message_df, reacts, title, participants) = parse_html_messages(directory)

(message_df, reacts, title, participants) = parse_json_messages(json_directory)
colour_palette = Category20[20][0:len(participants)]

mt_fig = create_message_timeseries_fig(message_df, title, participants, colour_palette)
react_fig = create_react_breakdown_panel(reacts, title, participants, colour_palette)

app.layout = html.Div([
    html.Datalist(id = 'chat-titles', children = [html.Option(id = value, value = key) for key, value in chat_titles.items()]),
    html.Div([
        html.Div(id = "header",
            children = [html.H1("Messenger Analysis", className = "title-text"),
            dcc.Input(
                id = 'chat-search', 
                placeholder = 'Search chats...', 
                list = 'chat-titles',
                type = "search",
                debounce = False,
                className = 'search-bar')],
            style = {
                'width': '100%',
                'margin': 'auto'
            })],
        style = {
            'margin': "1% 2.5%",
            'width': '100%'
        }),
    html.Br(style = {'clear': 'both'}),
    html.Div([
        dcc.Tabs(id="pages", value='timeline', children = [
            dcc.Tab(label = "Message Data", value = 'timeline'),
            dcc.Tab(label = 'Reacts Data', value = 'reacts'),
            dcc.Tab(label = "Message Log", value = "log")
        ]),
        html.Div(id = 'main-content', children = [
        ])], style = {"height": "50px", "width": "100%"}
    )], style = {'clear': 'both',
                 'position': 'relative',
                 'float': 'left',
                 'width': '100%'}
    )

@app.callback(Output('main-content', 'children'), Input('pages', 'value'), Input('chat-search', 'value'))
def switch_tabs(tab, chat_name):
    if chat_name not in chat_titles.keys():
        return dash.no_update

    json_directory = json_directories[chat_titles[chat_name]]
    # need to find a good way to update the page with new data.
    (message_df, reacts, title, participants) = parse_json_messages(json_directory)
    colour_palette = Category20[20][0:len(participants)]

    if tab == 'timeline':
        return create_message_timeseries_fig(message_df, title, participants, colour_palette)
    elif tab == 'reacts':
        return create_react_breakdown_panel(reacts, title, participants, colour_palette)
    elif tab == 'log':
        return create_message_log_panel()

if __name__ == '__main__':
    app.run_server(debug = True)
