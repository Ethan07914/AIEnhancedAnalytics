import logging
import os
import pandas as pd
import sys

def is_empty(data, desired_type):
    assert isinstance(data, desired_type)
    assert len(data) != 0

def file_exists(file_path):
    assert os.path.exists(file_path)

def no_records_lost(prev_file, curr_file):
    prev_df = pd.read_csv(prev_file)
    curr_df = pd.read_csv(curr_file)
    if len(prev_df) != len(curr_df) :
        error_message = f'{prev_file} has {len(prev_df)} records but {curr_file} has {len(curr_df)} records'
        logging.error(error_message)
        print(error_message)
        sys.exit()
    print(f'No records lost between {prev_file} and {curr_file}: PASSED')



