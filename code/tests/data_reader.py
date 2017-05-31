"""First test file of the data_reader function."""
import promethee as prom
import data_reader as dr


def test():
    """Test the data sets and data_reader module.

    The file contains :

        Alternatives
        #####
        81.6,12.6
        82.4,13
        83,12.8
        80.2,12.7
        81.6,11.9
        80.9,13.1
        80.9,12.2
        79.1,12.9
        82,13
        81.8,12.5
        83,10.6
        84,11.2
        80,11.8
        82.2,12.1
        80.7,13.1
        82.6,10.6
        81.9,11.9
        82.4,12.5
        81.7,11.7
        83.5,11.5
    """
    data_set = 'data/HDI/raw.csv'
    matrix = dr.open_raw(data_set)
    print(matrix[0])
