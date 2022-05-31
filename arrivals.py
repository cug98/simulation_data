import time
from datetime import datetime
import numpy as np
import pandas as pd

day_length = 60 * 60 * 24
resolution = 60 * 60  # resolution of arrival rates in seconds
day_start_time = 60 * 60 * 6  # timestamp of start of daytime in seconds, e.g. 0 for whole day, daytime>=day_start_time
day_end_time = 60 * 60 * 20  # timestamp of end of daytime in seconds, e.g. 60*60*24 for whole day, daytime<day_end_time
get_single_days = True

# suppress false warning
pd.options.mode.chained_assignment = None


def cleanup_data(raw_data):
    """ remove every passenger with empty leaving time from dataframe """
    # remove blanks
    raw_data.replace("", np.nan, inplace=True)
    raw_data.dropna(subset=['b5'], inplace=True)
    return raw_data


def add_weekly_normed_timestamps(raw_data):
    """ add extra column for weekly normed timestamp """

    # add column with weekday number
    raw_data['weekday'] = raw_data.apply(lambda row: datetime.fromtimestamp(get_timestamp(row, 'b1')).weekday(), axis=1)

    raw_data['hour'] = raw_data.apply(lambda row: datetime.fromtimestamp(get_timestamp(row, 'b1')).hour, axis=1)

    # time of arrival in week as seconds from Monday 12am
    raw_data['arrival_time'] = raw_data.apply(lambda row: get_weekly_normed_timestamp(row), axis=1)

    return raw_data


def get_timestamp(row, column_name):
    """ returns timestamp from given row and given column name """
    return time.mktime(datetime.strptime(row[column_name], '%d.%m.%Y %H:%M:%S').timetuple())


def get_weekly_normed_timestamp(row):
    """ return hour of day from given row and given column name """

    seconds_in_day = 60 * 60 * 24
    seconds_in_week = seconds_in_day * 7
    # epoch time starts on a Thursday, but we want 0 to equal monday, thus the addition of 3 days here
    return (time.mktime(
        datetime.strptime(row['b1'], '%d.%m.%Y %H:%M:%S').timetuple()) + 3 * seconds_in_day) % seconds_in_week


def analysis(df, time_name):
    """ output for basic data analysis stuff """
    types = ['economy', 'business']
    for i in types:
        f = open("arrival_rates_data_const_hourly.txt", "a")
        f.write('*' * 80 + '\n')
        f.write('Ankunftsraten fuer ' + i + ' ' + time_name + ':\n')
        df_type = df[df['type'] == i]
        for j in range(0, 23):
            f.write('Rate ab ' + str(j) + ' Uhr:' + str(
                len(df_type[(df_type['hour'] >= j) & (df_type['hour'] < j + 1)])) + '\n')
        f.write('\n\n')
    f.close()


def analysis_single_day(df):
    weekday = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for i in range(0, len(weekday)):
        data_frame_single_day = df[(df.weekday == i)]
        analysis(data_frame_single_day, weekday[i])


if __name__ == '__main__':
    # clear output txt files
    open("arrival_rates_data_const_hourly.txt", "w").close()

    data_frame = pd.read_csv('data.csv', sep=';')

    data_frame = cleanup_data(data_frame)
    data_frame = add_weekly_normed_timestamps(data_frame)
    if get_single_days:
        analysis_single_day(data_frame)
#    else:
# analysis_working_day_weekend(data_frame)
