import os

def is_empty(data, desired_type):
    assert isinstance(data, desired_type)
    assert len(data) != 0

def file_exists(file_path):
    assert os.path.exists(file_path)