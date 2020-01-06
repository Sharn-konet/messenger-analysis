# Messenger Analysis - An Interactive Visual Analysis Application

## Introduction:
Messenger Analysis is a pet project I've been developing in my spare time. Initially, the project was just an opportunity for me to familiarise myself with data visualisation techniques and with manipulating data. However, as the project has developed, the focus is now more on enabling others to familiarise themselves with the data that they openly give to larger companies.

To this end, my goals in developing this are:
* To give the user a comprehensive set of tools for analysing their data
* To make these tools intuitive to use, regardless of the user's proficiency in programming or data analytics
* To make the application robust, such that it can be used easily by a variety of users

## Setup:
Currently, to run this project, you need to include a Facebook Data query in the local directory. The only supported format is HTML.

The following command should then be run in the terminal:

```
bokeh serve --show messenger_analysis_bokeh.py
```