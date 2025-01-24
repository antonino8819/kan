# Performs KAN models decoding tests


from kan_decoder import run_decoder


# Set the seed for test reproducibility
seed = 10

# Set test name
typed_value = input("Insert test name ('base', 'simple', 'large', 'high_pruning, 'arch'):")
test_name = typed_value
while (test_name != 'base' and
       test_name != 'simple' and
       test_name != 'large' and
       test_name != 'high_pruning' and
       test_name != 'arch'):
    print("Wrong value inserted! Please, try again.\n --- \n")
    typed_value = input("Insert test name ('base', 'simple', 'large', 'high_pruning', 'arch'):")
    test_name = typed_value

# Test a simple KAN decoding
if test_name == 'base':  # a basic KAN
    run_decoder(2,
                2,
                [2,8,0],
                5,
                3,
                0.4,
                seed,
                f'{test_name}')

elif test_name == 'simple':  # a very simple KAN
    run_decoder(1,
                1,
                [1,0,0],
                5,
                3,
                0.4,
                seed,
                f'{test_name}')

elif test_name == 'large':  # a large KAN
    run_decoder(7,
                7,
                [10,10,0],
                5,
                3,
                0.4,
                seed,
                f'{test_name}')

elif test_name == 'high_pruning':  # a KAN subject to a high pruning (no continuity expected)
    run_decoder(2,
                2,
                [2, 8, 0],
                5,
                3,
                0.45,
                seed,
                f'{test_name}')

elif test_name == 'arch':  # test architecture list with 0 elements
    run_decoder(2,
                2,
                [0,2,0],
                5,
                3,
                0.2,
                seed,
                f'{test_name}')

