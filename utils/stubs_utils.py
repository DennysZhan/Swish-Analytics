import os
import pickle

def save_stub(stub_path, stub_data):
    """
    Saves the stub data to a file.

    Args:
        stub_path: Path to the stub file.
        stub_data: Data to be saved in the stub.
    """
    if not os.path.exists(os.path.dirname(stub_path)):
        os.makedirs(os.path.dirname(stub_path))

    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(stub_data, f)

def read_stub(read_from_stub, stub_path):
    """
    Reads the stub data from a file.

    Args:
        stub_path: Path to the stub file.

    Returns:
        Data read from the stub file.
    """
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
        
    return None
