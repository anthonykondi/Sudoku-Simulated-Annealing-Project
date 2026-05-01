import numpy as np
import matplotlib.pyplot as plt

### DEFINING VARIABLES ###

rng = np.random.default_rng()   # using 134 for testing

COORDS = [(i, j) for i in range(9) for j in range(9)]

### DEFINING FUNCIONS ###  

def get_box_coords(coord):
    q_row = coord[0] // 3
    q_col = coord[1] // 3
    box_coords = set([(3 * q_row + r_0, 3 * q_col + r_1) for r_0 in range(3) for r_1 in range(3)])
    return box_coords


def count_instances_rcb(board, coord):    # returns a dictionary of number of instances of numbers 1-9 in the row, column, and box of the input coord (includes the cell in count)
    instances = {i: 0 for i in range(1, 10)}

    # Creating the set of coordinates to check
    coords_to_check = set([])

    row_to_check = set([(coord[0], i) for i in range(9)])
    column_to_check = set([(i, coord[1]) for i in range(9)])
    box_to_check = get_box_coords(coord)

    coords_to_check.update(row_to_check, column_to_check, box_to_check)   # Set of 2-tuples (coordinates)

    # Counting occurences
    for xy in coords_to_check:
        xy_value = board[xy]
        if xy_value == 0:    # This will only get triggered for when generating the given board (0s denote empty cells)
            continue
        instances[xy_value] += 1

    return instances


def pattern(r,c):      # Function used as part generating a full valid board
    return (3*(r % 3) + r // 3 + c) % 9


def shuffle(s):        # Shuffles the elements in a container data type
    b = list(s)
    shuffled_s = []
    for _ in range(len(b)):
        idx = int(np.floor(rng.random() * len(b)))
        shuffled_s.append(b[idx])
        b.pop(idx)
    return shuffled_s


def set_given_board(n_givens):    # Generate a random board with 17 givens 
    # randomize rows, columns and numbers (of valid base pattern)
    rBase = range(3) 
    rows  = [g * 3 + r for g in shuffle(rBase) for r in shuffle(rBase)] 
    cols  = [g * 3 + c for g in shuffle(rBase) for c in shuffle(rBase)]
    nums  = shuffle(range(1, 10))

    # produce board using randomized baseline pattern
    board = np.array([[nums[pattern(r, c)] for c in cols] for r in rows])

    # remove cell values until we are left with desired board size
    leftover = 81 - n_givens
    given_coords = set([(i, j) for i in range(9) for j in range(9)])
    
    while leftover > 0:
        rand_x_coord = int(np.floor(rng.random() * 9))
        rand_y_coord = int(np.floor(rng.random() * 9))
        rand_coord = (rand_x_coord, rand_y_coord)
        
        if rand_coord not in given_coords:    
            continue

        given_coords.remove(rand_coord)
        board[rand_coord] = 0
        leftover -= 1
    
    given_state = (board, tuple(given_coords))   # convert given_coords set to a tuple to avoid making changes
    return given_state


def completely_random_fill(state):     # fill up the empty cells with random values (no smart rules)
    board, givens = state
    for coord in COORDS:
        if coord in givens: continue
        num = int(np.ceil(rng.random() * 9))
        board[coord] = num
    
    filled_state = (board, givens)
    return filled_state


def smart_fill(state):        # fill up the empty cells of each row so that there are no repeats
    board, givens = state
    for r in range(9):
        to_add = [n for n in range(1, 10) if n not in board[r, :]]
        to_add = shuffle(to_add)
        
        for c in range(9):      # fills the row with the random sequence 
            if (r, c) in givens: continue
            
            board[r, c] = to_add[0]
            to_add.pop(0)
    
    filled_state = (board, givens)
    return filled_state


def check_error(state, coord):
    board, givens = state
    if coord in givens: return 0 
    instances = count_instances_rcb(board, coord)
    cell_val = board[coord]
    if instances[cell_val] > 1:   # if the number occurs more than once it is an error
        return 1
    return 0


def change_t1(state):    # with this, it could not break below E = 14 consistently
    """
    Type 1: changes the state by going through the entire board cell-by-cell
    and flipping the cell value to the smalles from the computed {instances}. 
    """
    board, givens = state
    for coord in COORDS:
        if coord in givens: continue
        instances = count_instances_rcb(board, coord)
        min_num = 1000    # placeholder big number
        min_val = 1
        # update the cell to lowest occuring value in its rcb
        for val, num in instances.items():
            if num < min_num: 
                min_num, min_val = num, val
        board[coord] = min_val
    
    new_state = (board, givens)
    return new_state


def change_t2(state):    # This is terrible, gets to around E = 50, also doesn't change to a neighbouring config
    """
    Type 2: changes the state by randomly flipping values to other random values 
    """
    board, givens = state
    for coord in COORDS:
        if coord in givens: continue
        # 50% chance that it will flip that value to any other one (even to itself again)
        if rng.random() > 0.5:           
            new_val = int(np.ceil(rng.random() * 9))
            board[coord] = new_val
    
    new_state = (board, givens)
    return new_state


def change_t3(state):    # gets stuck at E = 3 to 8 because all the errors are between rows not within them, appears to decrease E quite fast
    """
    Type 3: changes the state by swapping elements in each row based on if they 
    are part of an error (pairwise swap for only one pair in each column).
    """
    board, givens = state
    for r in range(9):
        error_cells = [check_error(state, (r, c)) for c in range(9)]   # list of 0s and 1s 
        if sum(error_cells) > 1:
            err_idxs = [(r, c) for c in range(9) if error_cells[c] > 0]
            err_idxs = shuffle(err_idxs)    # randomizing the choice of indexes
            # performing the swap of cell values
            a = board[err_idxs[0]]
            b = board[err_idxs[1]]
            board[err_idxs[0]] = b
            board[err_idxs[1]] = a
    
    new_state = (board, givens)   # I don't think it is even necessary to create a new state variable 
    return new_state


def change_t4(state):
    """
    Type 4: changes the state by swapping elements in each row based on if they 
    are part of an error pairwise swap with kick for a single element in a row 
    (low probability for when it gets stuck).
    """
    board, givens = state
    for r in range(9):
        error_cells = [check_error(state, (r, c)) for c in range(9)]   # list of 0s and 1s 
        
        if sum(error_cells) > 1:
            err_idxs = [(r, c) for c in range(9) if error_cells[c] > 0]
            err_idxs = shuffle(err_idxs)    # randomizing the choice of indexes
            # performing the swap of cell values
            a = board[err_idxs[0]]
            b = board[err_idxs[1]]
            board[err_idxs[0]] = b
            board[err_idxs[1]] = a
        
        elif sum(error_cells) == 1:    # random kick for the single errors
            if 0.2 > rng.random():     # should be high, as it is most useful when it gets stuck in deep local minima
                other_cells = set([(r, c) for c in range(9) if error_cells[c] == 0])
                swap_cells = other_cells - set(givens)     # getting rid of the givens
                swap_cells = shuffle(swap_cells)

                err_idxs = [(r, c) for c in range(9) if error_cells[c] > 0]   # this is only one value (theres 100% an easier way to do this)
                err_idx = err_idxs[0]      # this is bad code but whatever, I'll change it eventually
                swap_idx = swap_cells[0]

                # performing the swap of cell values
                a = board[err_idx]
                b = board[swap_idx]
                board[err_idx] = b
                board[swap_idx] = a
    
    new_state = (board, givens)   # I don't think it is even necessary to create a new state variable 
    return new_state


def energy(state):      # counts number of cells that are part of an error
    E = 0
    for coord in COORDS:
        E += check_error(state, coord)
    return E


def choose_greedy(E_0, E_1):    # return True if new energy is lower than old energy
    return E_0 > E_1


def choose_metropolis(E_0, E_1, T):
    if T == 0:
        return choose_greedy(E_0, E_1)
    
    delta_E = E_1 - E_0
    if delta_E < 0: P_T_trans = 1

    else: P_T_trans = min(1, np.exp(-(E_1 - E_0) / T))
    # generate Unif(0,1) and check if it falls below transition probability
    return P_T_trans > rng.random()


def plot_E_T(E_history, T_history, t_max):
    t = [i for i in range(t_max + 1)]
    
    fig, ax = plt.subplots(2, 1)
    
    ax[0].plot(t, E_history); ax[0].set_title("Energy")
    ax[1].plot(t, T_history); ax[1].set_title("Temperature")

    plt.tight_layout()
    plt.show()


def linear_T(t, t_max, T_max=50):
    return T_max * (1 - t / t_max)


def multi_linear_T(t, t_max, T_max=50, n_cycles=4):
    return T_max * (1 - (t % int(np.ceil(t_max / n_cycles))) / (t_max / n_cycles))


def shrink_multi_linear_T(t, t_max, T_max=50, n_cycles=4):
    A = T_max / np.ceil((t + 1) * n_cycles / t_max)
    return A * (1 - (t % int(np.ceil(t_max / n_cycles))) / (t_max / n_cycles))


def exp_decay_T(t, t_max, T_max=50):
    return T_max * np.exp(- 5 * t / t_max)


def multi_exp_decay_T(t, t_max, T_max=50, n_cycles=4):
    return T_max * np.exp(- 5 * (t % int(np.ceil(t_max / n_cycles))) / (t_max / n_cycles))


def shrink_multi_exp_decay_T(t, t_max, T_max=50, n_cycles=4):
    A = T_max / np.ceil((t + 1) * n_cycles / t_max)
    return A * np.exp(- 5 * (t % int(np.ceil(t_max / n_cycles))) / (t_max / n_cycles))


def exp_T(t, t_max, T_max=50, n_cycles=4, frac=2):
    A = T_max / (frac ** (np.ceil((t + 1) * n_cycles / t_max) - 1))
    return A * np.exp(- 5 * (t % int(np.ceil(t_max / n_cycles))) / (t_max / n_cycles))


### GENERATE STARTING CONFIGURATION ###

t_max = int(100e1)
T_max = 50
n_givens = 17     # Must be >=17 for a unique solution

state = set_given_board(n_givens)   # create an empty sudoku board with n given numbers

print(state[0])

state = smart_fill(state)

E = energy(state)
t = 0

E_history = [E] 
T_history = [T_max]

### RUNNING THE ALGORITHM ###

progress = [n * t_max / 10 for n in range(1, 11)]

while E > 0 and t < t_max:
    T = exp_T(t, t_max, T_max=T_max, n_cycles=10, frac=1)
    
    if t in progress:      # progress updater
        print(f"PROGRESS = {100 * t / t_max}%")

    state_new = change_t4(state)
    E_new = energy(state_new)

    if choose_metropolis(E, E_new, T):
        state = state_new
        E = E_new

    E_history.append(E)
    T_history.append(T)
    t += 1
print("PROGRESS = 100.0%")

if E == 0:
    print(f"SOLVED: E = {E}")
    print(state[0])

plot_E_T(E_history, T_history, t)