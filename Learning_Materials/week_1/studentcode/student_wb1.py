from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list:
    """simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """

    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # create an empty candidate solution
    my_attempt = CandidateSolution()
    
    # ====> insert your code below here
    my_attempt.variable_values = []
    for i in puzzle.value_set:
        for j in puzzle.value_set:
            for k in puzzle.value_set:
                for l in puzzle.value_set:
                    my_attempt.variable_values = [i, j, k, l]
                    try:
                        result = puzzle.evaluate(my_attempt.variable_values)
                        if result:                        
                            return my_attempt.variable_values
                    except ValueError as e:
                        print(e)

    # <==== insert your code above here
    
    # should never get here
    return [-1, -1, -1, -1]

def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here
    for i in range(namearray.shape[0]):
        family_name_arr = namearray[i, -6:]
        family_name_str = ''
        family_name_str = family_name_str.join(family_name_arr)
        family_names.append(family_name_str)
    
    # <==== insert your code above here
    return family_names

def check_sudoku_array(attempt: np.ndarray) -> int:
    tests_passed = 0
    slices = []  # this will be a list of numpy arrays
    
    # ====> insert your code below here


    # use assertions to check that the array has 2 dimensions each of size 9
    arr_dim = (9, 9)
    assert attempt.shape == arr_dim, 'Incorrect array'

    ## Remember all the examples of indexing above
    ## and use the append() method to add something to a list
    for i in range(arr_dim[0]):
        slices.append(attempt[i, : ])
    for i in range(arr_dim[1]):
        slices.append(attempt[ : , i])
    for i in range(0, arr_dim[0], 3):
        for j in range(0, arr_dim[1], 3):
            slices.append(attempt[i:i+3, j:j+3])

    unique_values_no = 0
    for slice in slices:  # easiest way to iterate over list
        # print(slice) - useful for debugging?
        print(slice)
        # get number of unique values in slice
        unique_values_no = np.unique(slice).shape[0]
        # increment value of tests_passed as appropriate
        tests_passed += 1 if unique_values_no == 9 else 0
    print(tests_passed)
    
    # <==== insert your code above here
    # return count of tests passed
    return tests_passed
