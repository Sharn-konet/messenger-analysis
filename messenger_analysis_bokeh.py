#+=================================================+
#|           messenger_analysis_bokeh.py           |
#+=================================================+
# Author: Sharn-konet Reitsma

# Description:
# This script is an alternative implementation of my original messenger analysis script
# and it's corresponding functions. It uses XPath to parse the downloaded HTML files,
# which is much faster than its BeautifulSoup equivalent.

# In addition to this I've also implemented all plots into Bokeh, such that I can introduce
# more interactivity with the plots, giving people a bit more power to do their own analyses.

#--------------------------------------------------------------------------------------------
# SCRIPT START
#--------------------------------------------------------------------------------------------

from glob import glob
import scrapy
from datetime import datetime, timedelta, date
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, GroupFilter, CDSView, BoxAnnotation, Panel, Tabs, HoverTool
from bokeh.models.widgets import CheckboxButtonGroup
from bokeh.palettes import Spectral11, Category20, viridis
from bokeh.models.glyphs import MultiLine
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row, Spacer
from bokeh.models.widgets.sliders import DateRangeSlider
from bokeh.models.formatters import NumeralTickFormatter
from copy import deepcopy

#-------------------------------------------------------------------------
# Parsing Messenger HTML Files:
#-------------------------------------------------------------------------

directories = glob("**/messages/inbox/*/*.html", recursive = True)
directories = {directory.split("\\")[-2]: directory for directory in directories}
message_names = {key: key.split("_")[0] for key in directories}

directory = directories['THELOVECHAT_BT-aNw8Nzg']

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

data = scrapy.Selector(text = text, type = 'html')

title = data.xpath('//title/text()').extract()[0]

alternative = True

if alternative is False:
    names = data.xpath('//div[@class="' + attributes['names'] + '"]/text()').extract()
    participants = list(set(names))

    dates = scrapy.Selector(text=text, type='html').xpath('//div[@class="' + attributes['dates'] + '"]/text()').extract()
    dates = [datetime.strptime(date, "%b %d, %Y, %H:%M %p") for date in dates]

    reacts = data.xpath('//ul[@class="' + attributes['reacts'] + '"]/li/text()').extract()
    reacts = [(react[1:], react[0]) for react in reacts]
    reacts = pd.DataFrame(data = reacts, columns = ['Name', 'Reacts'])
    reacts = reacts.groupby(["React", "Name"])["React"].count()
    reacts.name = 'Count'
    reacts = reacts.reset_index()

    messages = data.xpath('//div[@class="' + attributes['message'] + '"]/div/div[2]//text()|//img[1]/@src').extract()
    # messages2 = scrapy.Selector(text=text, type ='html').xpath('//div[@class="' + attributes['message'] + '"]/div/div[2]/text()').extract()
    # messages3 = scrapy.Selector(text=text, type ='html').xpath('//img[@class="' + attributes['images'] + '"]').extract()
    # messages4 = scrapy.Selector(text=text, type ='html').xpath('//img/@src').extract()

    # lebndifferences = list(set(messages) - set(messages2))

    #-------------------------------------------------------------------------
    # Handle <br> tags:
    #-------------------------------------------------------------------------

    breaks = data.xpath('//div[@class="' + attributes['message'] + '"]/div/div[2]//br/../text()').extract()
    break_indices = [breaks.index(value) for value in breaks if value[0] != ' ']
    break_indices.append(len(breaks))

    for i in range(len(break_indices)-1):
        replace_index = messages.index(breaks[break_indices[i]])
        new_message = "\n".join(breaks[(break_indices[i]):break_indices[i+1]])
        messages[replace_index] = new_message
        del messages[replace_index + 1 : replace_index + break_indices[i+1] - break_indices[i]]

    # Create pandas dataframe from extracted datum:
    messages = list(zip(names, dates))
    df_master = pd.DataFrame(data = messages, columns = ['Name', 'Date'])
    for index in df_master.index:
            df_master.loc[index, 'Date'] = df_master.loc[index, 'Date'] - timedelta(hours = df_master.loc[index, 'Date'].hour, minutes = df_master.loc[index, 'Date'].minute)

    #messages = scrapy.Selector(text=text, type ='html').xpath('(//div[@class="' + attributes['message'] + '"]/div/div/text()|(//div[@class="' + attributes['message'] + '"]/div//a/@href)[1])').extract()

if alternative:
    boxes = data.xpath('//div[@class="pam _3-95 _2pi0 _2lej uiBoxWhite noborder"]')
    del boxes[0] # removes initial box which lists participants of the chat
    messages_structured = [0]*len(boxes)
    i = 0 

    participants = data.xpath('//div[@class="' + attributes['participants'] + '"]//text()').extract()[0].split(': ')[1].split(', ')
    participants.append(participants[-1].split(' and ')[1])
    participants[-2] = participants[-2].split(' and ')[0]

    for message in boxes:
        messages_structured[i] = {'Message': message.xpath('.//div[@class="' + attributes['message'] + '"]/div/div[2]//text()|.//audio/@src|.//a/@href').extract(),
                            'Date': message.xpath('.//div[@class="' + attributes['dates'] + '"]/text()').extract()[0],
                            'Reacts': message.xpath('.//ul[@class="' + attributes['reacts'] + '"]/li/text()').extract(),
                            'Name': message.xpath('.//div[@class="' + attributes['names'] + '"]/text()').extract()[0]}

        messages_structured[i]['Reacts'] = [(react[1:], react[0]) for react in messages_structured[i]['Reacts']]

        messages_structured[i]['Date'] = datetime.strptime(messages_structured[i]['Date'], "%b %d, %Y, %H:%M %p")

        # Should replace with resample to keep messages together and stuff.
        messages_structured[i]['Date'] = messages_structured[i]['Date'].replace(hour = 0, minute = 0)

        i += 1

    df_master = pd.DataFrame(messages_structured)

    reacts = list(df_master['Reacts'])
    reacts = [react[0] for react in reacts if len(react) > 0]
    reacts = pd.DataFrame(data = reacts, columns = ['Name', 'Reacts'])
    reacts = reacts.groupby(["Reacts", "Name"])["Reacts"].count()
    reacts.name = 'Count'
    reacts = reacts.reset_index()

#data[data['reacts'].apply(len) == max(data['reacts'].apply(len))]
## - Code which selects all messages which got the maximum number of reacts.
##   Can add an additional ['reacts'] to then see what the reacts are


#-------------------------------------------------------------------------
# Plot Data:
#-------------------------------------------------------------------------

# Find x-axis limits:
start_date = min(df_master.loc[:,'Date'])
end_date = max(df_master.loc[:,'Date'])

# Create widget objects:
name_buttons = CheckboxButtonGroup(labels = participants, active=[i for i in range(len(participants))])
date_slider = DateRangeSlider(end=end_date, start = start_date, value=(start_date, end_date), step = 1)

#Create a color palette to use in plotting:
# mypalette = viridis(len(participants))
mypalette=Category20[20][0:len(participants)]


# Create figures to be included:
p = figure(plot_width=800, plot_height=250, x_axis_type="datetime", toolbar_location = None)
p.toolbar.logo = None
p.x_range.start = start_date
p.x_range.end = end_date
p.toolbar.active_drag = None
p.toolbar.active_scroll = None

messages_tooltip = HoverTool(
    tooltips = [
        ('Name', '@Name'),
        ('Message Count', '@Message'),
        ('Date', '@Date') #{%D/%M/%Y}
    ],
    formatters = {
        'Date': 'datetime'
    }
)

p.add_tools(messages_tooltip)


p2 = figure(plot_height = 80, plot_width = 800, x_axis_type='datetime', toolbar_location=None,
           x_range=(start_date, end_date))
p2.yaxis.major_label_text_color = None
p2.yaxis.major_tick_line_color= None
p2.yaxis.minor_tick_line_color= None
p2.grid.grid_line_color=None
p2.toolbar.active_drag = None
p2.toolbar.active_scroll = None

box = BoxAnnotation(fill_alpha=0.5, line_alpha=0.5, level='underlay', left=start_date, right=end_date)

messages = df_master.groupby(['Name', 'Date']).count().reset_index()
messages = messages.loc[:, messages.columns != 'Reacts']

source = ColumnDataSource(data = messages)

# Plot a line for each person onto both figures:
for i in range(len(participants)):
    view=CDSView(source=source, 
    filters=[GroupFilter(column_name='Name', group=participants[i])])
    p.line(
        x='Date',
        y='Message',
        source=source,
        view=view,
        alpha=0.3,
        muted_color=mypalette[i], 
        muted_alpha=0.2,
        legend_label = participants[i],
        color = mypalette[i]
    )

    p.circle(
        x = 'Date',
        y = 'Message',
        alpha = 0.45,
        source = source,
        view = view,
        muted_color=mypalette[i], 
        muted_alpha=0.2,
        legend_label = participants[i],
        color = mypalette[i]
    )

    p2.line(
        x='Date',
        y='Message',
        source=source,
        view=view,
        color = mypalette[i]
    )

p.xaxis.axis_label = 'Time'
p.yaxis.axis_label = 'Total Messages'
p.title.text = title
p.legend.location = "top_right"
p.legend.click_policy="mute"

p2.add_layout(box)

#--------------------------------------------------------------------------+
# Creating Real Python callbacks for interaction between plots and widgets |
#--------------------------------------------------------------------------+

def subset_data(initial = False):
    df = df_master
    selected_names = [name_buttons.labels[i] for i in name_buttons.active]
    df = df[df['Name'].isin(selected_names)]
    df = df.groupby(by = ['Name', 'Date']).count().reset_index()
    return df

def update_graph(active_labels):
    print(active_labels)
    df = subset_data()
    source.data = dict(
        Date=df['Date'],
        Message=df['Message'],
        Name=df["Name"],
    )

def update_range(attr, old, new):
    #df = subset_data()
    start = datetime.fromtimestamp(new[0]/1e3)
    end = datetime.fromtimestamp(new[1]/1e3)
    #df = df[df['Date'] >= start]
    #df = df[df['Date'] <= end]
    p.x_range.start = start
    p.x_range.end = end

    box.left = start
    box.right = end

    # source.data = dict(
    #     Dates=df['Date'],
    #     Message=df['Message'],
    #     Names=df["Names"],
    # )

# Assign callbacks to appropriate widget interactions
name_buttons.on_click(update_graph)
date_slider.on_change('value', update_range)

date_slider_layout = row(Spacer(width = 46, height = 50, sizing_mode = "fixed"), date_slider, sizing_mode = "scale_width")

plots = column(p, date_slider_layout, p2, sizing_mode = "scale_width")

# Create the layout of the Bokeh application
message_timeseries = layout([
    [name_buttons],
    [plots]
], sizing_mode="scale_width")

message_panel = Panel(child = message_timeseries, title = 'Message Data')

#--------------------------------------------------------------------------+
# Plotting reactions
#--------------------------------------------------------------------------+
unique_reacts = reacts['Reacts'].unique()

reacts = reacts.pivot(index = 'Reacts', columns = 'Name', values = 'Count').fillna(0)
sums = reacts.sum(axis = 1)

for i in reacts.index:
    reacts.loc[i,:] = reacts.loc[i,:].apply(lambda x: (x/sums[i]))

reacts_source = ColumnDataSource(reacts)

p3 = figure(plot_width=800, plot_height=250, x_range = unique_reacts, y_range = [0, 1], toolbar_location = None)
p3.xaxis.major_label_text_font_size = "25pt"
p3.toolbar.active_drag = None
p3.toolbar.active_scroll = None

# configure so that Bokeh chooses what (if any) scroll tool is active

# p3.segment(0, "React", "Count", "React", line_width=2, line_color="green", source = reacts_source, )
# p3.circle("Count", "React", size=15, fill_color="orange", line_color="green", line_width=3, source = reacts_source)

# for i in range(len(participants)):
#     view=CDSView(source=reacts_source, 
#     filters=[GroupFilter(column_name='Names', group=participants[i])])
#     p3.segment(
#         x0 = "React",
#         y0 = 0,
#         x1 = "React",
#         y1 = "Count",
#         source = reacts_source,
#         view = view,
#         legend_label = participants[i],
#         color = mypalette[i]
#     )

#     p3.circle(
#         x = 'Reacts',
#         y = 'Count',
#         source = reacts_source,
#         view = view,
#         color = mypalette[i]
#     )

p3.vbar_stack(
    participants,
    x = "Reacts",
    width = 0.6,
    source = reacts_source,
    legend_label = participants,
    color = mypalette,
    fill_alpha = 0.5
)

p3.yaxis.formatter=NumeralTickFormatter(format = "0%")
legend = p3.legend[0]
legend.orientation = 'horizontal'
legend.location = 'center_right'
legend.spacing = 18
p3.add_layout(legend, 'above')

reacts_panel = layout([
    [p3]
], sizing_mode="scale_width")

reacts_panel = Panel(child = reacts_panel, title = 'Reacts Data')

tabs = Tabs(tabs = [message_panel, reacts_panel])

show(tabs)

curdoc().add_root(tabs)

show(message_timeseries)