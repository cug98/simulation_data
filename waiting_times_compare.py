import time
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

data_files = {
    "alle historisch": 'data.csv',
    "simuliert": 'sim_data.csv',
    "simuliert mit 1 business-WTMD": 'sim_data_1WTMD.csv',
}


def plot_and_save(datas_to_plot, title, x_label, y_label, filename, bins):
    fig, axs = plt.subplots(len(list(datas_to_plot)), 1, sharex='all', sharey='all')
    plt.suptitle(title)
    for i, key_element in enumerate(datas_to_plot):
        axs[i].set(title=key_element, xlabel=x_label, ylabel=y_label)
        # list.append()
        axs[i].hist(datas_to_plot[key_element], bins)

    # bins_np = np.histogram(np.hstack((data_to_plot, data_to_plot_sim)), bins=bins)[1]
    fig.tight_layout()

    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    folder = 'WaitingTimes/'
    plt.savefig(folder + filename)
    plt.show()


def cleanup_data(raw_data):
    """ remove every passenger with empty leaving time from dataframe """
    # remove blanks
    raw_data.replace("", np.nan, inplace=True)
    raw_data.dropna(subset=['b5'], inplace=True)
    return raw_data


def add_timestamps(raw_data):
    """ add new columns with unix timestamps in dataframe respective for all gates """
    # create new columns with timestamps
    raw_data['b1_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b1'), axis=1)
    raw_data['b2_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b2'), axis=1)
    raw_data['b3_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b3'), axis=1)
    raw_data['b4_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b4'), axis=1)
    raw_data['b5_timestamp'] = raw_data.apply(lambda row: to_timestamp(row, 'b5'), axis=1)

    return raw_data


def add_data_fields(raw_data):
    """ add extra columns like time for completion for every passenger """
    # boolean value whether day of arrival is weekday or not
    # raw_data['is_weekday'] = raw_data.apply(lambda row: is_weekday(row), axis=1)

    # add column with weekday number
    raw_data['weekday'] = raw_data.apply(lambda row: datetime.fromtimestamp(row['b1_timestamp']).weekday(), axis=1)

    # hour of arrival
    raw_data['arrival_time'] = raw_data.apply(lambda row: get_daytime(row), axis=1)

    # time diff for complete process
    raw_data['b1_b5_diff'] = (raw_data['b5_timestamp'] - raw_data['b1_timestamp'])

    # waiting times between checkpoints
    raw_data['b1_b2_diff'] = (raw_data['b2_timestamp'] - raw_data['b1_timestamp'])
    raw_data['b2_b3_diff'] = (raw_data['b3_timestamp'] - raw_data['b2_timestamp'])
    raw_data['b3_b4_diff'] = (raw_data['b4_timestamp'] - raw_data['b3_timestamp'])
    raw_data['b4_b5_diff'] = (raw_data['b5_timestamp'] - raw_data['b4_timestamp'])
    return raw_data


def to_timestamp(row, column_name):
    """ returns timestamp from given row and given column name """
    return time.mktime(datetime.strptime(row[column_name], '%d/%m/%Y %H:%M:%S').timetuple())


def get_daytime(s):
    """ return hour of day from given row and given column name """
    return datetime.fromtimestamp(s['b1_timestamp']).hour


def get_basic_analysis(data, type_name):
    """ do some basic data analysis from given dataset like max waiting time, min waiting time and mean waiting time"""
    f = open("data_analysis_dump.txt", "a")
    f.write('*' * 80 + '\n')
    f.write('Basic analysis for ' + type_name + ':\n')
    f.write('max: ' + str(max(data['b1_b5_diff']) / 60) + '\n')
    f.write('min: ' + str(min(data['b1_b5_diff']) / 60) + '\n')
    f.write('mean: ' + str(statistics.mean(data['b1_b5_diff']) / 60) + '\n')
    f.write('standard deviation: ' + str(statistics.stdev(data['b1_b5_diff']) / 60) + '\n\n')
    f.close()


def plot_waiting_times(dfs, type_name):
    """ plot distribution of waiting times between checkpoints"""
    for i in range(1, 5):
        df_diffs = {}
        for key_element in dfs:
            df_diffs[key_element] = dfs[key_element]['b' + str(i) + '_b' + str(i + 1) + '_diff']
            df_diffs[key_element] = [x / 60 for x in df_diffs[key_element]]

        plot_and_save(df_diffs,
                      title='Wartezeit zwischen ' + 'b' + str(i) + ' und b' + str(i + 1) + ' für ' + type_name,
                      y_label='Anzahl', x_label='Wartezeit[min]',
                      filename='Wartezeit zwischen ' + 'b' + str(i) + ' und b' + str(
                          i + 1) + ' für ' + type_name + '.png', bins=100)
    df_diffs = {}
    for key_element in dfs:
        df_diffs[key_element] = dfs[key_element]['b1_b5_diff']
        df_diffs[key_element] = [x / 60 for x in df_diffs[key_element]]

    plot_and_save(df_diffs, title='Wartezeit zwischen b1 und b5' + ' für ' + type_name,
                  y_label='Anzahl', x_label='Wartezeit[min]',
                  filename='Wartezeit zwischen b1 und b5 für ' + type_name + '.png', bins=100)


def analyze_waiting_times(df, type_name):
    """ get data analysis for waiting time between checkpoints"""
    f = open("waiting_times.txt", "a")
    f.write('*' * 80 + '\n')
    f.write('Wartezeiten fuer ' + type_name + ':\n')
    for i in range(1, 5):
        df_diff = df['b' + str(i) + '_b' + str(i + 1) + '_diff']
        df_diff = [x / 60 for x in df_diff]
        f.write('zwischen ' + 'b' + str(i) + ' und b' + str(i + 1) + '\n')
        f.write('min: ' + str(min(df_diff)) + '\n')
        f.write('max: ' + str(max(df_diff)) + '\n')
        f.write('Durchschnitt: ' + str(statistics.mean(df_diff)) + '\n')
        f.write('Standardabweichung: ' + str(statistics.stdev(df_diff)) + '\n\n')
    df_diff = df['b1''_b5''_diff']
    df_diff = [x / 60 for x in df_diff]
    f.write('zwischen ' + 'b1 und b5''\n')
    f.write('min: ' + str(min(df_diff)) + '\n')
    f.write('max: ' + str(max(df_diff)) + '\n')
    f.write('Durchschnitt: ' + str(statistics.mean(df_diff)) + '\n')
    f.write('Standardabweichung: ' + str(statistics.stdev(df_diff)) + '\n\n')
    f.close()


def do_stuff(df, time_name):
    """ output for basic data analysis stuff """
    get_basic_analysis(df[df['type'] == 'economy'], 'economy ' + time_name)
    get_basic_analysis(df[df['type'] == 'business'], 'business ' + time_name)
    get_basic_analysis(df, 'all ' + time_name)


def do_stuff_single_day(df):
    weekday = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for i in range(0, len(weekday)):
        data_frame_single_day = df[df.weekday == i]
        do_stuff(data_frame_single_day, weekday[i])


if __name__ == '__main__':
    # clear output txt files
    open("waiting_times.txt", "w").close()

    Path("WaitingTimes/").mkdir(parents=True, exist_ok=True)
    all_df = {}
    for key in data_files:
        all_df[key] = pd.read_csv(data_files[key], sep=';')

    for key in all_df:
        all_df[key] = cleanup_data(all_df[key])
        all_df[key] = add_timestamps(all_df[key])
        all_df[key] = add_data_fields(all_df[key])

    plot_waiting_times(all_df, 'alle')
    for key in all_df:
        analyze_waiting_times(all_df[key], key)
        print('SLA ' + key + ':', str(len(all_df[key][all_df[key].b1_b5_diff <= (60 * 30)]) / len(all_df[key]))[0: 8])
