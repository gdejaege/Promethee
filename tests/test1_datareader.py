"""First test file of the data_reader function."""
import promethee as prom
import data_reader as dr


def test():
    """Test the datasets and data_reader module."""
    data_set = 'data/EPI/raw.csv'
    matrix = dr.open_raw(data_set)
    print(matrix[2])
