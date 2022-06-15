import time
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statistics import mean

data_files = {
    "historische Daten": 'sim_data/data.csv',
    "simuliert": 'sim_data/sim_data.csv',
    "simuliert mit Passagieranstieg": 'sim_data/sim_data_pass_inc.csv',
    "simuliert mit Systemausfall": 'sim_data/sim_data_sys_failure.csv',
    "Verbesserung": 'sim_data/sim_data_1late_1WTMD_3BC.csv',
    "Verbesserung mit Passagieranstieg": 'sim_data/sim_data_1late_1WTMD_pass_inc_3BC.csv',
    "Verbesserung mit Systemausfall": 'sim_data/sim_data_1late_1WTMD_3BC_sys_fail.csv',
}
# for count of passengers in system
time_step_size_passengers = 60 * 5

# for time window during which SLA is examined
time_step_size_SLA = 60 * 30

# for time window during which average waiting times is examined
time_step_size_means = 60 * 10

SLA_time = 60 * 30
business_only = False


def plot_and_save_waiting_times(datas_to_plot, title, x_label, y_label, filename, bins):
    plt.rcParams.update({'figure.figsize': (7, 9), 'figure.dpi': 1000})
    fig, axs = plt.subplots(len(list(datas_to_plot)), 1, sharex='all', sharey='all')
    plt.suptitle(title)

    list_for_bins = []
    for key_element in datas_to_plot:
        list_for_bins = np.hstack((datas_to_plot[key_element], list_for_bins))
    bins_np = np.histogram(list_for_bins, bins=bins)[1]

    for i, key_element in enumerate(datas_to_plot):
        axs[i].set(title=key_element, xlabel=x_label, ylabel=y_label)
        axs[i].hist(datas_to_plot[key_element], bins_np)

    fig.tight_layout()

    folder = 'WaitingTimes/'
    plt.savefig(folder + filename)
    # plt.show()


def plot_and_save_passengers_in_system(datas_to_plot, x_label, y_label, title, filename):
    plt.rcParams.update({'figure.figsize': (7, 9), 'figure.dpi': 1000})
    fig, axs = plt.subplots(len(list(datas_to_plot)), 1, sharex='all', sharey='all')
    plt.suptitle(title)
    for i, key_element in enumerate(datas_to_plot):
        axs[i].set(title=key_element, xlabel=x_label, ylabel=y_label)
        axs[i].bar(range(0, 60 * 60 * 24 * 7, time_step_size_passengers), datas_to_plot[key_element],
                   width=time_step_size_passengers)

    fig.tight_layout()

    folder = 'CountPassengers/'
    plt.savefig(folder + filename)
    # plt.show()


def plot_and_save_average_waiting_times(datas_to_plot, x_label, y_label, title, filename):
    plt.rcParams.update({'figure.figsize': (7, 9), 'figure.dpi': 1000})
    fig, axs = plt.subplots(len(list(datas_to_plot)), 1, sharex='all', sharey='all')
    plt.suptitle(title)
    for i, key_element in enumerate(datas_to_plot):
        axs[i].set(title=key_element, xlabel=x_label, ylabel=y_label)
        axs[i].bar(range(0, 60 * 60 * 24 * 7, time_step_size_means), datas_to_plot[key_element],
                   width=time_step_size_means)

    fig.tight_layout()

    folder = 'AverageWaitingTimes/'
    plt.savefig(folder + filename)
    # plt.show()


def plot_and_save_sla(datas_to_plot, x_label, y_label, title, filename):
    plt.rcParams.update({'figure.figsize': (7, 9), 'figure.dpi': 1000})
    fig, axs = plt.subplots(len(list(datas_to_plot)), 1, sharex='all', sharey='all')
    plt.suptitle(title)
    for i, key_element in enumerate(datas_to_plot):
        axs[i].set(title=key_element, xlabel=x_label, ylabel=y_label)
        axs[i].plot(range(0, 60 * 60 * 24 * 7, time_step_size_SLA), datas_to_plot[key_element])
        axs[i].hlines(y=0.9, xmin=0, xmax=60 * 60 * 24 * 7, linewidth=2, color='r', label='SLA')

    fig.tight_layout()

    folder = 'SLA/'
    plt.savefig(folder + filename)
    # plt.show()


def cleanup_data(raw_data):
    """ remove every passenger with empty leaving time from dataframe """
    # remove blanks
    raw_data.replace("", np.nan, inplace=True)
    raw_data.dropna(subset=['b5'], inplace=True)
    if business_only:
        raw_data = raw_data[raw_data.type == 'business']
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

        plot_and_save_waiting_times(df_diffs,
                                    title='Wartezeit zwischen ' + 'b' + str(i) + ' und b' + str(
                                        i + 1) + ' für ' + type_name,
                                    y_label='Anzahl', x_label='Wartezeit[min]',
                                    filename='Wartezeit zwischen ' + 'b' + str(i) + ' und b' + str(
                                        i + 1) + ' für ' + type_name + '.png', bins=100)
    df_diffs = {}
    for key_element in dfs:
        df_diffs[key_element] = dfs[key_element]['b1_b5_diff']
        df_diffs[key_element] = [x / 60 for x in df_diffs[key_element]]

    plot_and_save_waiting_times(df_diffs, title='Wartezeit zwischen b1 und b5' + ' für ' + type_name,
                                y_label='Anzahl', x_label='Wartezeit[min]',
                                filename='Wartezeit zwischen b1 und b5 für ' + type_name + '.png', bins=100)


def plot_passengers_in_system(dfs, type_name, number_of_weeks):
    numbers_by_time = {}
    # iterate over all seconds within a week with step size of one hour
    for key_elements in dfs:
        numbers_by_time[key_elements] = []
        for i in range(0, 60 * 60 * 24 * 7, time_step_size_passengers):
            numbers_by_time[key_elements].append(len(dfs[key_elements][
                                                         (dfs[key_elements].b1_timestamp % (60 * 60 * 24 * 7) <= i) & (
                                                                 dfs[key_elements].b5_timestamp % (
                                                                 60 * 60 * 24 * 7) > i)]) // number_of_weeks)

    plot_and_save_passengers_in_system(numbers_by_time, y_label='Anzahl', x_label='Systemzeit[s]',
                                       title="Anzahl Passagiere in System für " + type_name,
                                       filename=type_name + '.png')


def plot_average_waiting_times(dfs, type_name):
    means_by_time = {}
    # iterate over all seconds within a week with step size of one hour
    for key_elements in dfs:
        means_by_time[key_elements] = []
        for i in range(0, 60 * 60 * 24 * 7, time_step_size_means):
            list_all_relevant_passengers = dfs[key_elements][
                (dfs[key_elements].b5_timestamp % (60 * 60 * 24 * 7) >= i) & (
                        dfs[key_elements].b5_timestamp % (60 * 60 * 24 * 7) < i + time_step_size_means)]['b1_b5_diff']
            try:
                means_by_time[key_elements].append(mean(list_all_relevant_passengers) / 60)
            except statistics.StatisticsError:
                # if not possible to calculate waiting time use last value
                means_by_time[key_elements].append(means_by_time[key_elements][-1])

    plot_and_save_average_waiting_times(means_by_time, y_label='Wartezeit[min]', x_label='Systemzeit[s]',
                                        title="Durchschnittliche Wartezeit für " + type_name,
                                        filename=type_name + '.png')


def plot_SLA(dfs, type_name):
    numbers_by_time = {}
    # iterate over all seconds within a week with step size of one hour
    for key_elements in dfs:
        numbers_by_time[key_elements] = []
        for i in range(0, 60 * 60 * 24 * 7, time_step_size_SLA):
            # percentage of passengers within SLA
            try:
                numbers_by_time[key_elements].append(len(dfs[key_elements][
                                                             (dfs[key_elements].b5_timestamp % (
                                                                     60 * 60 * 24 * 7) > i) & (
                                                                     dfs[key_elements].b5_timestamp % (
                                                                     60 * 60 * 24 * 7) <= i + time_step_size_SLA) & (
                                                                     dfs[key_elements].b1_b5_diff <= SLA_time)]) / len(
                    dfs[key_elements][(dfs[key_elements].b5_timestamp % (60 * 60 * 24 * 7) > i) & (
                            dfs[key_elements].b5_timestamp % (60 * 60 * 24 * 7) <= i + time_step_size_SLA)]))
            except ZeroDivisionError:
                numbers_by_time[key_elements].append(0)
    plot_and_save_sla(numbers_by_time, y_label='Anzahl', x_label='Systemzeit[s]',
                      title="SLA für " + type_name,
                      filename=type_name + '.png')


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
    Path("CountPassengers/").mkdir(parents=True, exist_ok=True)
    Path("AverageWaitingTimes/").mkdir(parents=True, exist_ok=True)
    Path("SLA/").mkdir(parents=True, exist_ok=True)
    all_df = {}
    for key in data_files:
        all_df[key] = pd.read_csv(data_files[key], sep=';')

    for key in all_df:
        all_df[key] = cleanup_data(all_df[key])
        all_df[key] = add_timestamps(all_df[key])
        all_df[key] = add_data_fields(all_df[key])

    print('plotting waiting means...')
    plot_average_waiting_times(all_df, 'alle')
    print('plotting waiting times...')
    plot_waiting_times(all_df, 'alle')
    print('plotting passenger counts...')
    plot_passengers_in_system(all_df, 'alle', 3)
    print('plotting SLA...')
    plot_SLA(all_df, 'alle')

    print('analyzing data...')
    for key in all_df:
        analyze_waiting_times(all_df[key], key)

    outfile = open("waiting_times.txt", "a")
    outfile.write('\n\n')
    outfile.write('*' * 80 + '\n')
    for key in all_df:
        outfile.write('SLA ' + key + ':' + str(len(all_df[key][all_df[key].b1_b5_diff <= SLA_time]) / len(all_df[key]))[
                                           0: 8] + '\n')
        print('SLA ' + key + ':', str(len(all_df[key][all_df[key].b1_b5_diff <= SLA_time]) / len(all_df[key]))[0: 8])
    outfile.close()
