import glob
import os
import re
import shutil

import pandas as pd

from ClassificationUtils import load_csv_from_folder


def split_data_into_weeks(src_folder, dst_folder):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = f"{src_folder}/*.csv"  # DataBases/
    abs_path = os.path.join(script_dir, rel_path)

    all_files = glob.glob(abs_path)
    all_files.sort()

    files_to_dump = []
    set = 0

    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    for filename in all_files:
        files_to_dump.append(filename)

        if len(files_to_dump) == 7 or filename == all_files[-1]:
            set_folder_path = dst_folder + "\\Week" + str(set)
            os.mkdir(set_folder_path)
            set += 1

            for file in files_to_dump:
                shutil.copy2(file, set_folder_path)

            files_to_dump = []


def get_all_devices_file(file_path: str) -> list:

    header = list(pd.read_csv(file_path).columns)
    header.remove("timestamp")
    header.remove("smartMeter")
    return header


def contains_subdirectory_with_os_listdir(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            return True
    return False


def get_all_devices_data(folder: str, check_sub_folders=True) -> list:

    devices = set()

    script_dir = os.path.dirname(__file__)
    rel_path = f"{folder}/*.csv"
    files_path = os.path.join(script_dir, rel_path)

    for file in glob.glob(files_path):
        devices.update(get_all_devices_file(file))

    if check_sub_folders and contains_subdirectory_with_os_listdir(folder):
        rel_path = f"{folder}/*/"
        folders_path = os.path.join(script_dir, rel_path)
        folders = glob.glob(folders_path)
        for folder in folders:
            devices.update(get_all_devices_data(folder))

    return list(devices)


def exclude_unused_devices(src_folder, dst_folder):
    script_dir = os.path.dirname(__file__)
    series_rel_path = f"{src_folder}/TimeSeriesData/*"
    action_rel_seq_path = f"{src_folder}/ActionSeq/*"
    active_phases_rel_path = f"{src_folder}/Active_phases/*"

    series_path = os.path.join(script_dir, series_rel_path)
    all_weeks = sorted(glob.glob(series_path), key=lambda s: int(re.search(r'\d+', s).group()))

    # collect data of series
    series_data = [[] for i in range(len(all_weeks))]
    series_file_names = [[] for i in range(len(all_weeks))]

    for w in range(len(all_weeks)):
        files = glob.glob(all_weeks[w] + "/*")

        for file in files:
            series_data[w].append(pd.read_csv(file, index_col="timestamp"))
            series_file_names[w].append(os.path.basename(file))

    devices = get_all_devices_data(series_path[0:-1])

    # collect data of active phases
    active_phases_path = os.path.join(script_dir, active_phases_rel_path)
    all_weeks_active_phases = sorted(glob.glob(active_phases_path), key=lambda s: int(re.search(r'\d+', s).group()))

    active_phases_data = [[] for i in range(len(all_weeks_active_phases))]
    active_phases_file_names = [[] for i in range(len(all_weeks_active_phases))]

    for w in range(len(all_weeks_active_phases)):
        files = glob.glob(all_weeks_active_phases[w] + "/*")

        for file in files:
            active_phases_data[w].append(pd.read_csv(file, index_col="timestamp"))
            active_phases_file_names[w].append(os.path.basename(file))

    # check whether devices are used
    device_used = [False] * len(devices)

    for w in range(len(active_phases_data)):
        for f in range(len(active_phases_data[w])):
            # data = series_data[w][f]
            data = active_phases_data[w][f]
            devices_of_data = list(data.columns)
            # devices_of_data.remove("timestamp")
            # devices_of_data.remove("smartMeter")

            for d in range(len(devices)):
                if devices[d] in devices_of_data:
                    if not ((data[devices[d]] == 0).all()):
                        device_used[d] = True

    # delete unused devices out of dataframes
    for w in range(len(series_data)):
        for f in range(len(series_data[w])):
            data = series_data[w][f]
            devices_of_data = list(data.columns)
            # devices_of_data.remove("timestamp")
            devices_of_data.remove("smartMeter")

            a_p_data = active_phases_data[w][f]
            active_phases_devices = list(a_p_data.columns)
            # active_phases_devices.remove("timestamp")

            for d in range(len(devices)):
                if devices[d] in devices_of_data:
                    if not device_used[d]:
                        data.drop(columns=[devices[d]], inplace=True)

                if devices[d] in active_phases_devices:
                    if not device_used[d]:
                        a_p_data.drop(columns=[devices[d]], inplace=True)

    # save modified TimeSeries data
    dst_series_path = os.path.join(script_dir, f"{dst_folder}/TimeSeriesData")

    if not os.path.exists(dst_series_path):
        os.mkdir(dst_series_path)

    for w in range(len(series_data)):
        filepath_week = dst_series_path + "/Week" + str(w)

        if not os.path.exists(filepath_week):
            os.mkdir(filepath_week)

        for f in range(len(series_data[w])):
            series_data[w][f].to_csv(filepath_week + "/" + series_file_names[w][f])

    # save modified active phases data
    dst_active_phases_path = os.path.join(script_dir, f"{dst_folder}/Active_phases")

    if not os.path.exists(dst_active_phases_path):
        os.mkdir(dst_active_phases_path)

    for w in range(len(active_phases_data)):
        filepath_week = dst_active_phases_path + "/Week" + str(w)

        if not os.path.exists(filepath_week):
            os.mkdir(filepath_week)

        for f in range(len(active_phases_data[w])):
            active_phases_data[w][f].to_csv(filepath_week + "/" + active_phases_file_names[w][f])

    # copy files from ActionSequences src to ActionSequences dst
    action_sequence_path = os.path.join(script_dir, action_rel_seq_path)
    all_weeks_action_sequences = sorted(glob.glob(action_sequence_path), key=lambda s: int(re.search(r'\d+', s).group()))

    dst_action_sequences_path = os.path.join(script_dir, f"{dst_folder}/ActionSeq")

    if not os.path.exists(dst_action_sequences_path):
        os.mkdir(dst_action_sequences_path)

    for w in range(len(all_weeks_action_sequences)):
        files = glob.glob(all_weeks_action_sequences[w] + "/*")
        filepath_week = dst_action_sequences_path + "/Week" + str(w)

        if not os.path.exists(filepath_week):
            os.mkdir(filepath_week)

        for file in files:
            shutil.copy2(file, filepath_week)


if __name__ == '__main__':
    # src_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
    #               + "\\Resources\\TimeSeriesData\\TimeSeriesData\\ActionSeq_active_phases")
    # dst_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
    #               + "\\Resources\\TimeDataWeeks\\Active_phases")
    # split_data_into_weeks(src_folder, dst_folder)
    # filepath = (os.path.dirname(os.path.abspath(os.path.curdir))
    #             + "\\Resources\\TimeSeriesData\\TimeSeriesData\\2022-12-05.csv")
    # print(get_all_devices_file(filepath))
    exclude_unused_devices(os.path.dirname(os.path.abspath(os.path.curdir)) + "\\Resources\\TimeDataWeeks",
                           os.path.dirname(os.path.abspath(os.path.curdir)) + "\\Resources\\TimeDataWeeksOnlyUsedDevices")
    # print(get_all_devices_data(os.path.dirname(os.path.abspath(os.path.curdir)) + "/Resources/TimeDataWeeks/TimeSeriesData"))

