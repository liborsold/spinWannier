# import the LinearChain class from the models module within the fuNEGF package in the parallel directory
# print the directories in path

from spinWannier.wannier_utils import parse_KPOINTS_file
import numpy as np

def test_parse_KPOINTS_file_1():
    """
    Test the parsing of a KPOINTS files.
    """
    
    # Test 1
    kpoint_matrix, Nkpoints, kpath_ticks = parse_KPOINTS_file("tests/test_files/KPOINTS_parse_1")
    kpoint_matrix_expected = [
        [(0.3333333333,  0.3333333333,  0.000000),    (0.00,  0.00,  0.00)],
        [(0.00,  0.00,  0.00),    (0.50, 0.00,  0.00)],
        [(0.50, 0.00,  0.00),    (0.3333333333,  0.3333333333,  0.000000)]
    ]
    Nkpoints_expected = 51
    kpath_ticks_expected = ['K', 'G', 'M', 'K']

    assert np.all(
        np.isclose(kpoint_matrix, kpoint_matrix_expected)
    ), "KPOINTS matrix is not parsed correctly"
    assert Nkpoints == Nkpoints_expected, "Number of kpoints is not parsed correctly"
    assert kpath_ticks == kpath_ticks_expected, "KPOINTS path ticks are not parsed correctly"


def test_parse_KPOINTS_file_2():
    """
    Test the parsing of a KPOINTS files.
    """

    # Test 2
    kpoint_matrix, Nkpoints, kpath_ticks = parse_KPOINTS_file("tests/test_files/KPOINTS_parse_2")
    kpoint_matrix_expected = [
        [(0.3333333333,  0.3333333333,  0.000000),    (0.00,  0.00,  0.00)],
        [(0.50, 0.00,  0.00),    (0.3333333333,  0.3333333333,  0.000000)], 
        [(0, 0, 0), (0, 0, 0)],
        [(0, 0, 0), (0, 0, 0)]
    ]
    Nkpoints_expected = 101
    kpath_ticks_expected = ['K', 'G|M', 'K|', '', '']


    assert np.all(
        np.isclose(kpoint_matrix, kpoint_matrix_expected)
    ), "KPOINTS matrix is not parsed correctly"
    assert Nkpoints == Nkpoints_expected, "Number of kpoints is not parsed correctly"
    assert kpath_ticks == kpath_ticks_expected, "KPOINTS path ticks are not parsed correctly"
