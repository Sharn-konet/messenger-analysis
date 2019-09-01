from bs4 import BeautifulSoup
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

with open("C:\\Users\\sharn\\Documents\\Coding Projects\\Python\\messenger_analysis\\messages\\inbox\\JadeHunter_rzWb1rO8vw\\message_1.html", encoding = "utf8") as fp:
    soup = BeautifulSoup(fp, "lxml")

messages = soup.find_all(class_ = "_3-96 _2pio _2lek _2lel")

dates = []

for message in messages:
    if 'Sharn-konet Reitsma' in message.contents:
        dates.append(datetime.strptime(message.next_sibling.next_sibling.contents[0], '%b %d, %Y, %I:%M %p'))

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

for i in range(len(dates_set)):
    element = dates_set.pop()
    message_count.append(dates.count(element))
    date_final_list.append(element)

index_table = np.argsort(date_final_list)
date_final_list = np.array(date_final_list)[index_table]
message_count = np.array(message_count)[index_table]

plt.plot_date(date_final_list, message_count, 'b-')
#plt.xlim(737122, 737296)
plt.show()