import glob
import os
import shutil

import pandas as pd


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


def get_all_devices(file_path: str) -> list:

    header = list(pd.read_csv(file_path).columns)
    header.remove("timestamp")
    header.remove("smartMeter")
    return header


if __name__ == '__main__':
    # src_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
    #               + "\\Resources\\TimeSeriesData\\TimeSeriesData\\ActionSeq_active_phases")
    # dst_folder = (os.path.dirname(os.path.abspath(os.path.curdir))
    #               + "\\Resources\\TimeDataWeeks\\Active_phases")
    # split_data_into_weeks(src_folder, dst_folder)
    filepath = (os.path.dirname(os.path.abspath(os.path.curdir))
                + "\\Resources\\TimeSeriesData\\TimeSeriesData\\2022-12-05.csv")
    print(get_all_devices(filepath))
