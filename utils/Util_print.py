import time
import os


def print_elapsed_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time of {func.__name__}: {elapsed_time}s")
        return result

    return wrapper


def print_separator(atom_character: str = "=") -> None:
    assert type(atom_character) is str, "The type of atom_character should be string."
    assert len(atom_character) == 1, "The length of atom_character should be 1"

    size = os.get_terminal_size()
    width = size.columns
    print(atom_character * width)
