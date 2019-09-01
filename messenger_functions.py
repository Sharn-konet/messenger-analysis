from bs4 import BeautifulSoup
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def get_dates(folder_name, names = ['Sharn-konet Reitsma']):
    """ Retreives the dates of each message from a Facebook generated HTML file

        Parameters:
        -----------

        folder_name: string
            A string which gives the name of the folder which identifies the message chain.

        names: list
            A list which contains all the names of messages you want to count.

        Returns:
        ---------

        date_dictionary: dictionary
            Stores a list of times which messages were messaged, organised by names.

        Notes:
        -------

    """

    # Retrieves the html file in the form of a soup object
    with open("messages\\inbox\\" + folder_name + "\\message_1.html", encoding = "utf8") as fp:
        soup = BeautifulSoup(fp, "lxml")

    # Finds all of the elements with the message CSS class
    messages = soup.find_all(class_ = "_3-96 _2pio _2lek _2lel")


    # Goes through the messages for each name, storing the dates in a list, and then a dictionary.
    date_dictionary = {}

    for name in names:
        dates = []
        character_sum = 0
        word_count = 0
        for message in messages:
            if name in message.contents:
                dates.append(datetime.strptime(message.next_sibling.next_sibling.contents[0], '%b %d, %Y, %I:%M %p'))
                character_sum += len(message.next_sibling.contents[0].contents[1].text)
                word_count += len(message.next_sibling.contents[0].contents[1].text.split())

        date_dictionary[name] = {"dates":dates, "characters":character_sum, "wordcount":word_count}

    return date_dictionary

def count_messages(dates):
    """ From a list of dates, counts the occurences of each day in the list.

        Parameters:
        -----------

        dates: datetime
            List of datetime objects of each message recieved.

        Returns:
        ---------

        date_final_list: list
            List of each date that is included in the search

        message_count: list
            A list of integers indexed to match date_final_list. 
            Tally of messages on each day.

        Notes:
        -------
    """

    dates = mdates.date2num(dates)
    dates = np.floor(dates)
    dates = dates.tolist()

    max_date = int(max(dates))
    min_date = int(min(dates))

    dates_set = set()
    for date in range(min_date, max_date):
        dates_set.add(date)

    message_count = []
    date_final_list = []
    
    for _ in range(len(dates_set)):
        element = dates_set.pop()
        message_count.append(dates.count(element))
        date_final_list.append(element)
    
    index_table = np.argsort(date_final_list)
    date_final_list = np.array(date_final_list)[index_table]
    message_count = np.array(message_count)[index_table]

    return date_final_list, message_count

def get_proportion(data, names):
    """ From a list of dates, counts the occurences of each day in the list.

        Parameters:
        -----------

        data: dictionary
            Dictionary containing Dates and corresponding message tallies, with names as the key.

        Notes:
        -------
    """
    total_messages = []

    for name in names:
        total_messages.append(sum(data[name][1]))

    plt.pie(total_messages, labels = names, autopct='%1.1f%%')
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)

    plt.axis('equal')
    plt.title("Proportion of Total Messages by Name")
    plt.show()

if __name__ == "__main__":
    pass