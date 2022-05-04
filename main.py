import time
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitter import Fitter, get_distributions


def plot_and_save(data_to_plot, title, x_label, y_label, filename, bins, fit_dist=False):
    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    plt.hist(data_to_plot, bins=bins)

    if fit_dist:
        # stuff to fit distribution to data
        # all available distributions in AnyLogic
        anylogic_distributions = ['bernoulli', 'beta', 'beta (truncated)', 'binomial', 'binomial (truncated)', 'cauchy',
                                  'chi2', 'erlang', 'exponential', 'exponential (truncated)', 'gamma',
                                  'gamma (truncated)', 'geometric', 'gumbel1', 'gumbel2', 'hypergeometric', 'laplace',
                                  'logarithmic', 'logistic', 'lognormal', 'negativeBinomial',
                                  'negativeBinomial (truncated)', 'normal', 'normal (truncated)', 'pareto', 'pert',
                                  'poisson', 'poisson (truncated)', 'randomFalse', 'randomTrue', 'rayleigh',
                                  'triangular', 'triangular (truncated)', 'triangularAV', 'uniform', 'uniform_discr',
                                  'uniform_pos', 'weibull', 'weibull (truncated)']
        # TODO: use only distributions available in AnyLogic
        fitter = Fitter(data_to_plot, distributions=get_distributions(), timeout=60)
        fitter.fit()
        fitter.summary(Nbest=3)
        # save information about distribution fitting in txt file
        f = open("fitting_distribution_data.txt", "a")
        f.write('*' * 80 + '\n')
        f.write('fitter info for ' + filename + ':\n')
        f.write(str(fitter.get_best()) + '\n\n')
        f.close()

    plt.gca().set(title=title, ylabel=y_label, xlabel=x_label)
    plt.savefig('Images/' + filename)
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
    return time.mktime(datetime.strptime(row[column_name], '%d.%m.%Y %H:%M:%S').timetuple())


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


def plot_waiting_time_complete(df, type_name):
    """ plot distribution of complete waiting time for given dataframe """
    df_diff = [x / 60 for x in df['b1_b5_diff']]
    plot_and_save(df_diff, title='Verteilung Wartezeit ' + type_name, y_label='Occurrences', x_label='Wartezeit[min]',
                  filename='Verteilung Wartezeit ' + type_name + '.png', bins=250)


def plot_arrivals(df, type_name):
    """ plot arrival rate of given dataframe by daytime"""
    df_arrivals = [x for x in df['arrival_time']]
    plot_and_save(df_arrivals, title='Ankunft ' + type_name, y_label='Occurrences', x_label='Uhrzeit[h]',
                  filename='Ankunft ' + type_name + '.png', bins=24)


def plot_waiting_times(df, type_name):
    """ plot distribution of waiting times between checkpoints"""
    for i in range(1, 5):
        df_diff = df['b' + str(i) + '_b' + str(i + 1) + '_diff']
        df_diff = [x / 60 for x in df_diff]
        plot_and_save(df_diff, title='Wartezeit zwischen ' + 'b' + str(i) + ' und b' + str(i + 1) + ' für ' + type_name,
                      y_label='Wartezeit', x_label='Wartezeit[min]',
                      filename='Wartezeit zwischen ' + 'b' + str(i) + ' und b' + str(
                          i + 1) + ' für ' + type_name + '.png', bins=100, fit_dist=True)
    df_diff = df['b1_b5_diff']
    df_diff = [x / 60 for x in df_diff]
    plot_and_save(df_diff, title='Wartezeit zwischen b1 und b5' + ' für ' + type_name,
                  y_label='Wartezeit', x_label='Wartezeit[min]',
                  filename='Wartezeit zwischen b1 und b5 für ' + type_name + '.png', bins=100, fit_dist=True)


def analyze_waiting_times(df, type_name):
    """ get data analysis for waiting time between checkpoints"""
    f = open("data_analysis_dump.txt", "a")
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
    plot_waiting_time_complete(df[df['type'] == 'economy'], 'economy ' + time_name)
    plot_waiting_time_complete(df[df['type'] == 'business'], 'business ' + time_name)

    plot_arrivals(df[df['type'] == 'economy'], 'economy ' + time_name)
    plot_arrivals(df[df['type'] == 'business'], 'business ' + time_name)

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
    open("data_analysis_dump.txt", "w").close()
    open("fitting_distribution_data.txt", "w").close()

    data_frame = pd.read_csv('data.csv', sep=';')

    data_frame = cleanup_data(data_frame)
    data_frame = add_timestamps(data_frame)
    data_frame = add_data_fields(data_frame)

    # all weekdays
    # data_frame_weekday = data_frame[data_frame.weekday < 5]
    # do_stuff(data_frame_weekday, 'weekday')

    # for weekends
    # data_frame_weekend = data_frame[data_frame.weekday >= 5]
    # do_stuff(data_frame_weekend, 'weekend')

    # do_stuff(data_frame, 'complete')
    plot_waiting_times(data_frame, 'alle')
    analyze_waiting_times(data_frame, 'alle')

    # do_stuff_single_day(data_frame)
