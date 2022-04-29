import time
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_timestamps(raw_data):
    # remove blanks
    raw_data.replace("", np.nan, inplace=True)
    raw_data.dropna(subset=['b5'], inplace=True)

    # create new columns with timestamps
    raw_data['b1_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b1'), axis=1)
    raw_data['b2_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b2'), axis=1)
    raw_data['b3_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b3'), axis=1)
    raw_data['b4_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b4'), axis=1)
    raw_data['b5_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b5'), axis=1)
    raw_data['is_weekday'] = raw_data.apply(lambda row: is_weekday(row), axis=1)
    raw_data['arrival_time'] = raw_data.apply(lambda row: get_daytime(row), axis=1)

    # time diff for complete process
    raw_data['b1_b5_diff'] = (raw_data['b5_timestamp'] - raw_data['b1_timestamp'])

    return raw_data


def to_timestamp(s, column_name):
    return time.mktime(datetime.strptime(s[column_name], '%d.%m.%Y %H:%M:%S').timetuple())


def is_weekday(s):
    return datetime.fromtimestamp(s['b1_timestamp']).weekday() < 5


def get_daytime(s):
    return datetime.fromtimestamp(s['b1_timestamp']).hour


def get_basic_analysis(data, type_name):
    print(type_name, 'max: ', max(data['b1_b5_diff']) / 60)
    print(type_name, 'min: ', min(data['b1_b5_diff']) / 60)
    print(type_name, 'mean: ', statistics.mean(data['b1_b5_diff']) / 60)
    print(type_name, 'standard deviation: ', statistics.stdev(data['b1_b5_diff']) / 60)


def plot_waiting_times(df, type_name):
    df_diff = [x / 60 for x in df['b1_b5_diff']]
    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    plt.hist(df_diff, bins=500)
    plt.gca().set(title='Verteilung ' + type_name, ylabel='Occurrences')
    plt.show()


def plot_arrivals(df, type_name):
    df_diff = [x for x in df['arrival_time']]
    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    plt.hist(df_diff, bins=24)
    plt.gca().set(title='Ankunft ' + type_name, ylabel='Occurrences')
    plt.show()


def do_stuff(df, time_name):
    # plot_waiting_times(df[df['type'] == 'economy'], 'economy ' + time_name)
    # plot_waiting_times(df[df['type'] == 'business'], 'business ' + time_name)

    plot_arrivals(df[df['type'] == 'economy'], 'economy ' + time_name)
    plot_arrivals(df[df['type'] == 'business'], 'business ' + time_name)

    get_basic_analysis(df[df['type'] == 'economy'], 'economy ' + time_name)
    get_basic_analysis(df[df['type'] == 'business'], 'business ' + time_name)
    get_basic_analysis(df, 'all ' + time_name)


if __name__ == '__main__':
    data_frame = pd.read_csv('data.csv', sep=';')

    data_frame = add_timestamps(data_frame)

    print(len(data_frame))
    data_frame_weekday = data_frame[data_frame.is_weekday]
    do_stuff(data_frame_weekday, 'weekday')

    data_frame_weekend = data_frame[data_frame.is_weekday == False]
    do_stuff(data_frame_weekend, 'weekend')

    do_stuff(data_frame, 'complete')
    print(len(data_frame))
