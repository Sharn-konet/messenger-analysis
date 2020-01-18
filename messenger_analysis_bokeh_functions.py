from multiprocessing import Pool
import scrapy
import json
import itertools
import pandas as pd
from datetime import datetime

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
                print("god help us all")
                message['Message'] = None
                message['Type'] = None

        message['Name'] = message.pop('sender_name')

        message['Date'] = datetime.fromtimestamp(message.pop('timestamp_ms')/1000)
        message['Date'] = message['Date'].replace(hour=0, minute=0, second = 0, microsecond = 0)

        if 'reactions' in message:
            message['Reacts'] = [(react['actor'], react['reaction']) for react in message['reactions']]
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