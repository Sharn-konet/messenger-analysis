# +==========================================================+
# |           messenger_analysis_data_functions.py           |
# +==========================================================+
# Author: Sharn-konet Reitsma

# Description:
# A collection of functions which are used to parse and handle the data which is visualised.
# This file also includes callbacks which are used outside of the panels themselves, ie. when
# switch between different chats

import scrapy
import json
import itertools
import pandas as pd
from datetime import datetime
from glob import glob
from functools import partial

from bokeh.palettes import Category20
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Tabs, AutocompleteInput, Div
from bokeh.layouts import column, row, Spacer

from messenger_analysis_panels import create_message_timeseries_fig, create_react_breakdown_panel, create_message_log_panel, create_individual_statistics_panel

def get_chat_titles(directories, key):
    directory = directories[key][0]

    with open(directory.replace("\\", "/"), encoding = 'utf-8') as dataFile:
        message_data = json.load(dataFile)

    title = message_data['title']

    return title


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
        
        message_info = {*message.keys()} - {'sender_name', 'timestamp_ms', 'type', 'reactions', 'share', 'content', 'users'}

        message['Details'] = None

        # Doesn't make sense to use a for/else statement here. Usually only one type / no support for multiple
        if message['type'] == 'Call':
            message['Details'] = message['call_duration']
            message['Type'] = 'Call'
        
        else:
            for key in message_info:
                value = message.pop(key)
                message['Message'] = [data['uri'] if type(value) is list else str([*value.values()][0]) for data in value]
                message['Type'] = key.capitalize()
                break

            else:
                if 'users' in message.keys():
                    name = None
                    content = message.pop('content')
                    ## NEEDS SUPPORT FOR REMOVING MEMBERS FROM THE GROUP (couldn't find example)
                    if message['type'] == 'Subscribe':
                        name = content.split('added')[1]
                        name = name.strip()[:-14]
                    message['Message'] = content
                    message['Type'] = message['type']
                    message['Details'] = name
                else:
                    try:
                        message['Message'] = message.pop('content').encode('latin-1').decode('utf-8')
                        message['Type'] = 'Message'
                    except KeyError:
                        message['Message'] = None
                        message['Type'] = 'Removed Message'

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
    
    if 'missed' in message_df.keys():
        message_df.loc[pd.notna(message_df['missed']), 'Type'] = 'Missed Call'

    keys_to_remove = set(message_df.columns) - {'Details', 'Message', 'Type', 'Name', 'Date', 'Reacts'}

    for key in keys_to_remove:
        del message_df[key]

    reacts_list = [*message_df['Reacts']]
    reacts = [react for reacts in reacts_list for react in reacts if len(reacts) > 0]

    reacts = pd.DataFrame(data=reacts, columns=['Name', 'Reacts'])
    reacts = reacts.groupby(["Reacts", "Name"])["Reacts"].count()
    reacts.name = 'Count'
    reacts = reacts.reset_index()

    return (message_df, reacts, title, participants)

def create_document(directory, chat_titles):

    (message_df, reacts, title, participants) = parse_json_messages(directory)

    # -------------------------------------------------------------------------
    # Plot Message Timeseries:
    # -------------------------------------------------------------------------

    # Create a color palette to use in plotting:
    """ Might raise an error when number of people in the group chat is > 20"""
    colour_palette = Category20[20][0:len(participants)]

    message_panel = create_message_timeseries_fig(message_df, title, participants, colour_palette)

    # --------------------------------------------------------------------------+
    # Plot Reaction Panel:
    # --------------------------------------------------------------------------+

    reacts_panel = create_react_breakdown_panel(reacts, title, participants, colour_palette)

    # --------------------------------------------------------------------------+
    # Create Panel to Summarise Individual Statistics:
    # --------------------------------------------------------------------------+

    message_log_panel = create_message_log_panel(message_df, title, participants, colour_palette)

    # --------------------------------------------------------------------------+
    # Compile Bokeh Application:
    # --------------------------------------------------------------------------+

    tabs = Tabs(tabs=[message_panel, reacts_panel, message_log_panel])

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

    return layout

def update_data(attr, old, new):
    """ Callback which allows for the document to be updated to a new group
    """

    json_directories = glob("**/messages/inbox/*/*.json", recursive = True)
    unique_chatnames = {*map(lambda dir: dir.split("\\")[-2], json_directories)}
    json_directories = {name: [directory for directory in json_directories if name in directory] for name in unique_chatnames}

    chat_titles = {get_chat_titles(json_directories, key): key for key in json_directories}
    
    if new in [*chat_titles]:

        new_directory = json_directories[chat_titles[new]]

        new_root = create_document(new_directory, chat_titles)

        curdoc().add_root(new_root)

        curdoc().remove_root(curdoc().roots[0])

"""
def inital_setup(format):

    if format == "JSON":
        json_directories = glob("**/messages/inbox/*/*.json", recursive = True)
        unique_chatnames = {*map(lambda dir: dir.split("\\")[-2], json_directories)}
        json_directories = {name: [directory for directory in json_directories if name in directory] for name in unique_chatnames}
        chat_titles = {get_chat_titles(json_directories, key): key for key in json_directories}
        directory = html_directories['THELOVECHAT_BT-aNw8Nzg']

    elif format == "HTML":
        html_directories = glob("**/messages/inbox/*/*.html", recursive = True)
        html_directories = {directory.split("\\")[-2]: directory for directory in html_directories}
        # need to check if JSOn and HTML give same output
        chat_titles = {name: parse_html_title(directory) for name, directory in html_directories.items()}
        message_names = {key: key.split("_")[0] for key in html_directories}
        directory = json_directories['THELOVECHAT_BT-aNw8Nzg']
    else:
        break
    layout = create_document(json_directory, chat_titles)

    curdoc().title = "Messenger Analysis"

    curdoc().add_root(layout)
"""