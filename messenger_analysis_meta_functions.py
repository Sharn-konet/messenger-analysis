import json
from glob import glob
from bokeh.palettes import Spectral11, Category20, viridis
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, GroupFilter, CDSView, BoxAnnotation, Panel, Tabs, HoverTool, AutocompleteInput
from messenger_analysis_bokeh_functions import parse_json_messages, create_message_timeseries_panel, create_react_breakdown_panel, create_individual_statistics_panel

def get_chat_titles(directories, key):
    directory = directories[key][0]

    with open(directory.replace("\\", "/"), encoding = 'utf-8') as dataFile:
        message_data = json.load(dataFile)

    title = message_data['title']

    return title

def update_data(attr, old, new):
    """ Callback which allows for the document to be updated to a new group
    """

    json_directories = glob("**/messages/inbox/*/*.json", recursive = True)
    unique_chatnames = {*map(lambda dir: dir.split("\\")[-2], json_directories)}
    json_directories = {name: [directory for directory in json_directories if name in directory] for name in unique_chatnames}

    chat_titles = {get_chat_titles(json_directories, key): key for key in json_directories}
    
    if new in [*chat_titles]:

        new_directory = json_directories[chat_titles[new]]

        (message_df, reacts, title, participants) = parse_json_messages(new_directory)

        colour_palette = Category20[20][0:len(participants)]

        message_panel = create_message_timeseries_panel(message_df, title, participants, colour_palette)

        reacts_panel = create_react_breakdown_panel(reacts, title, participants, colour_palette)

        individual_statistics_panel = create_individual_statistics_panel(message_df, title, participants, colour_palette)

        tabs = Tabs(tabs=[message_panel, reacts_panel, individual_statistics_panel])

        directory_search = AutocompleteInput(completions = list(chat_titles.keys()), width = 400, height = 30, sizing_mode = "fixed")
        directory_search.on_change("value", update_data)

        page = column(directory_search, tabs, sizing_mode = "scale_both")

        curdoc().clear()

        curdoc().add_root(page)