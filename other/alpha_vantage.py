# This code is an addendum to RNN_project.py
# It was used to obtain the data in the file IBM_full_daily_adjusted.csv
# It was run before the main program, and the resulting .csv was hosted publicly on GitHub

import requests
import csv
API_URL = "https://www.alphavantage.co/query"
data = {
    "function": "TIME_SERIES_DAILY",
    "symbol": "IBM",
    "outputsize": "full",
    "datatype": "csv",
    "apikey": "RSK6LEE9ZJTEVCXC"
    }
response = requests.get(API_URL, params=data)
decoded_response = response.content.decode('utf-8')
cr = csv.reader(decoded_response.splitlines(),delimiter=',')
my_list = list(cr)
with open('IBM_full_daily_adjusted.csv', 'w', newline='') as csvfile:
	mywriter = csv.writer(csvfile, delimiter=',')
	for row in my_list:
		mywriter.writerow(row)
