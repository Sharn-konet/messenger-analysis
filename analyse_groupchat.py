from messenger_functions import *

folder_name = 'THELOVECHAT_BT-aNw8Nzg'
names = ['Felix Walton', 'Ruby Griffin', 'Donald Mayo', 'Eamonn Tee', 'Max Griffin', 'Max Pitto', 'Sharn-konet Reitsma', 'Will Savoy', 'Zarina Hewlett']

date_dictionary = get_dates(folder_name, names)

data = {}

for name in names:
    date_final_list, message_count = count_messages(date_dictionary[name]["dates"])
    data[name] = [date_final_list, message_count]

# Plot all available names
for name in names:
    plt.plot_date(*data[name], '-', label = name)

plt.xlabel("Date")
plt.ylabel("Message Count")
plt.title("Message Tally Across All Time")
plt.legend()
plt.show()

get_proportion(data, names)

word_count = [date_dictionary[name]["wordcount"] for name in names]

plt.bar(range(len(names)), height = word_count, tick_label = names, width = 0.5)
plt.show()
