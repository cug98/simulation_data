import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitter import Fitter
from pathlib import Path

day_length = 60 * 60 * 24
resolution = 60 * 15  # resolution of arrival rates in seconds
day_start_time = 60 * 60 * 6  # timestamp of start of daytime in seconds, e.g. 0 for whole day, daytime>=day_start_time
day_end_time = 60 * 60 * 20  # timestamp of end of daytime in seconds, e.g. 60*60*24 for whole day, daytime<day_end_time
get_single_days = True

# suppress false warning
pd.options.mode.chained_assignment = None


# TODO: change x-axis for nighttime such that it displays the accurate times
def plot_and_save(data_to_plot, title, x_label, y_label, filename, bins, fit_dist=True):
    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    test_data = [(i / (60 * 60)) % 24 for i in data_to_plot]
    folder = 'Distribution_plots/' if fit_dist else 'Images/'

    if fit_dist:
        # stuff to fit distribution to data
        dist_in_both = ["beta", "cauchy", "chi2", "erlang", "expon", "truncexpon", "gamma", "gumbel_l", "gumbel_r",
                        "laplace", "loggamma", "loglaplace", "loguniform", "logistic", "lognorm", "norm", "truncnorm",
                        "pareto", "rayleigh", "triang", "uniform", "weibull_min", "weibull_max"]
        # fitter = Fitter(data_to_plot, distributions=dist_in_both, timeout=60, bins=bins)
        fitter = Fitter(test_data, distributions=dist_in_both, timeout=60, bins=bins)
        fitter.fit()
        fitter.summary(Nbest=3)
        # save information about distribution fitting in txt file
        f = open("fitting_distribution_arrivals_data.txt", "a")
        f.write('*' * 80 + '\n')
        f.write('fitter info for ' + filename + ':\n')
        f.write(str(fitter.get_best()) + '\n\n')
        f.close()

    plt.gca().set(title=title, ylabel=y_label, xlabel=x_label)
    plt.savefig(folder + filename)
    plt.show()


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


def plot_arrivals(df, type_name, bins=24):
    """ plot arrival rate of given dataframe by daytime"""
    df_arrivals = [x for x in df['arrival_time']]
    plot_and_save(df_arrivals, title='Ankunft ' + type_name, y_label='Occurrences', x_label='Uhrzeit[h]',
                  filename='Ankunft ' + type_name + '.png', bins=bins, fit_dist=True)


def analysis(df, time_name, bins=24):
    """ output for basic data analysis stuff """
    # if looking at nighttime, times have to be shifted such that 23:00 is next to 00:00
    if 'night' in time_name:
        df['arrival_time'] = df.apply(lambda row: ((row['arrival_time'] + (day_length - day_end_time)) % day_length),
                                      axis=1)

    plot_arrivals(df[df['type'] == 'economy'], 'economy ' + time_name, bins=bins)
    plot_arrivals(df[df['type'] == 'business'], 'business ' + time_name, bins=bins)


def analysis_working_day_weekend(df):
    data_frame_working_day_day = df[
        (df.weekday < 5) & ((df.arrival_time % day_length) >= day_start_time) & (
                (df.arrival_time % day_length) < day_end_time)]
    data_frame_working_day_night = df[((df.weekday < 5) & ((df.arrival_time % day_length) >= day_end_time)) | (
            (df.weekday < 6) & ((df.arrival_time % day_length) < day_start_time))]

    data_frame_weekend_day = df[
        ((df.weekday > 4) & ((df.arrival_time % day_length) >= day_start_time)) & (
                (df.arrival_time % day_length) < day_end_time)]
    data_frame_weekend_night = df[((df.weekday > 4) & ((df.arrival_time % day_length) >= day_end_time)) | (
            (df.weekday > 5) & ((df.arrival_time % day_length) < day_start_time))]
    bins_day = (day_end_time - day_start_time) // resolution
    bins_night = (60 * 60 * 24 - (day_end_time - day_start_time)) // resolution

    analysis(data_frame_working_day_day, 'working day daytime', bins=bins_day)
    analysis(data_frame_working_day_night, 'working day night', bins=bins_night)
    analysis(data_frame_weekend_day, 'weekend daytime', bins=bins_day)
    analysis(data_frame_weekend_night, 'weekend night', bins=bins_night)


def analysis_single_day(df):
    weekday = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for i in range(0, len(weekday)):
        data_frame_single_day_day = df[
            (df.weekday == i) & ((df.arrival_time % day_length) >= day_start_time) & (
                    (df.arrival_time % day_length) < day_end_time)]
        data_frame_single_day_night = df[((df.weekday == i) & ((df.arrival_time % day_length) >= day_end_time)) | (
                (df.weekday == (i + 1) % len(weekday)) & ((df.arrival_time % day_length) < day_start_time))]
        bins_day = (day_end_time - day_start_time) // resolution
        bins_night = (60 * 60 * 24 - (day_end_time - day_start_time)) // resolution
        analysis(data_frame_single_day_day, weekday[i] + ' daytime', bins=bins_day)
        analysis(data_frame_single_day_night, weekday[i] + ' nighttime', bins=bins_night)


if __name__ == '__main__':
    # clear output txt files
    open("arrival_rates_data.txt", "w").close()
    open("fitting_distribution_arrivals_data.txt", "w").close()

    Path("Images/").mkdir(parents=True, exist_ok=True)
    Path("Distribution_plots/").mkdir(parents=True, exist_ok=True)

    data_frame = pd.read_csv('data.csv', sep=';')

    data_frame = cleanup_data(data_frame)
    data_frame = add_weekly_normed_timestamps(data_frame)
    if get_single_days:
        analysis_single_day(data_frame)
    else:
        analysis_working_day_weekend(data_frame)
