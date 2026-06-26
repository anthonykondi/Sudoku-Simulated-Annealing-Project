import numpy as np
import matplotlib.pyplot as plt
import copy

### DEFINING VARIABLES ###

rng = np.random.default_rng()   # using 134 for testing

COORDS = [(i, j) for i in range(9) for j in range(9)]

### DEFINING FUNCIONS ###  

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


def set_given_board(n_givens):    # Generate a random board with n givens 
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


class SudokuBoard:

    COORDS = [(i, j) for i in range(9) for j in range(9)]

    def __init__(self, board: np.ndarray, given_coords: tuple):
        self._board = board                 # numpy.array, empty cells have zeros
        self._given_coords = given_coords   # tuple of tuples, coordinates of initially filled cells (non-zero)


    def __str__(self):
        return str(self._board)    # could make this prettier in the future using ascii art
    

    def __getitem__(self, coord):
        return self._board[coord]
    

    def __setitem__(self, coord, val):
        if val not in range(10):
            raise TypeError("Invalid sudoku entry")
        self._board[coord] = val


    def get_box_coords(coord):
        q_row = coord[0] // 3
        q_col = coord[1] // 3
        box_coords = set([(3 * q_row + r_0, 3 * q_col + r_1) for r_0 in range(3) for r_1 in range(3)])
        return box_coords


    def _count_instances_rcb(self, coord):    # returns a dictionary of number of instances of numbers 1-9 in the row, column, and box of the input coord (includes the cell in count)
        instances = {i: 0 for i in range(1, 10)}

        # Creating the set of coordinates to check
        coords_to_check = set([])

        row_to_check = set([(coord[0], i) for i in range(9)])
        column_to_check = set([(i, coord[1]) for i in range(9)])
        box_to_check = SudokuBoard.get_box_coords(coord)    # for the sake of encapsulation

        coords_to_check.update(row_to_check, column_to_check, box_to_check)   # Set of 2-tuples (coordinates)

        # Counting occurences
        for xy in coords_to_check:
            xy_value = self._board[xy]
            if xy_value == 0:    # This will only get triggered for when generating the given board (0s denote empty cells)
                continue
            instances[xy_value] += 1

        return instances


    def shuffle(s):          # Shuffles the elements in a container data type
        b = list(s)
        shuffled_s = []
        for _ in range(len(b)):
            idx = int(np.floor(rng.random() * len(b)))
            shuffled_s.append(b[idx])
            b.pop(idx)
        return shuffled_s


    def _smart_fill(self):    # fill up the empty cells of each row so that there are no repeats
        for r in range(9):
            to_add = [n for n in range(1, 10) if n not in self._board[r, :]]
            to_add = SudokuBoard.shuffle(to_add)   # using the one in the class for the sake of encapsulation
            
            for c in range(9):      # fills the row with the random sequence 
                if (r, c) in self._given_coords: continue
                
                self._board[r, c] = to_add[0]
                to_add.pop(0)   


    def _check_error(self, coord):
        if coord in self._given_coords: return 0 
        instances = self._count_instances_rcb(coord)
        cell_val = self._board[coord]
        if instances[cell_val] > 1:   # if the number occurs more than once it is an error
            return 1
        return 0


    def energy(self):      # counts number of cells that are part of an error
        E = 0
        for coord in SudokuBoard.COORDS:
            E += self._check_error(coord)
        return E


    def exp_T(t, t_max, T_max=50, n_cycles=4, frac=2):
        A = T_max / (frac ** (np.ceil((t + 1) * n_cycles / t_max) - 1))
        return A * np.exp(- 5 * (t % int(np.ceil(t_max / n_cycles))) / (t_max / n_cycles))


    def _choose_metropolis(E_0, E_1, T):
        if T == 0:
            return E_0 > E_1
        
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


    def _special_change(self):
        """
        Changes the state by swapping elements in each row based on if they 
        are part of an error pairwise swap with kick for a single element in a row 
        (low probability for when it gets stuck).
        """
        for r in range(9):
            error_cells = [self._check_error((r, c)) for c in range(9)]   # list of 0s and 1s 
            
            if sum(error_cells) > 1:
                err_idxs = [(r, c) for c in range(9) if error_cells[c] > 0]
                err_idxs = SudokuBoard.shuffle(err_idxs)    # randomizing the choice of indexes
                # performing the swap of cell values
                a = self._board[err_idxs[0]]
                b = self._board[err_idxs[1]]
                self._board[err_idxs[0]] = b
                self._board[err_idxs[1]] = a
            
            elif sum(error_cells) == 1:    # random kick for the single errors
                if 0.2 > rng.random():     # should be high, as it is most useful when it gets stuck in deep local minima
                    other_cells = set([(r, c) for c in range(9) if error_cells[c] == 0])
                    swap_cells = other_cells - set(self._given_coords)     # getting rid of the givens
                    swap_cells = SudokuBoard.shuffle(swap_cells)

                    err_idxs = [(r, c) for c in range(9) if error_cells[c] > 0]   # this is only one value (theres 100% an easier way to do this)
                    err_idx = err_idxs[0]      # this is bad code but whatever, I'll change it eventually
                    swap_idx = swap_cells[0]

                    # performing the swap of cell values
                    a = self._board[err_idx]
                    b = self._board[swap_idx]
                    self._board[err_idx] = b
                    self._board[swap_idx] = a


    def anneal_solve(self, plot_energy=True):
        """Solve sudoku using simulated annealing algorithm"""
        
        # starting off by resetting the board to the initial configuration
        for coord in SudokuBoard.COORDS:
            if coord not in self._given_coords:
                self._board[coord] = 0
        
        self._smart_fill()      # filling up the board according to smart fill rule (obey 1 of the 3 groups)
        
        # initiating variables
        t_max = int(100e1)
        T_max = 50
        E = self.energy()
        t = 0
        E_history = [E] 
        T_history = [T_max]

        # running the simulated annealing algorithm 
        while E > 0 and t < t_max:
            T = SudokuBoard.exp_T(t, t_max, T_max=T_max, n_cycles=10, frac=1)

            proposed_board = SudokuBoard(copy.deepcopy(self._board), self._given_coords)
            proposed_board._special_change()
            E_new = proposed_board.energy()

            if SudokuBoard._choose_metropolis(E, E_new, T):
                self._board = copy.deepcopy(proposed_board._board)
                E = E_new

            E_history.append(E)
            T_history.append(T)
            t += 1
        
        if plot_energy:
            SudokuBoard.plot_E_T(E_history, T_history, t)

        return True


### GENERATE STARTING CONFIGURATION ###

# t_max = int(100e1)
# T_max = 50
# n_givens = 17     # Must be >=17 for a unique solution

# state = set_given_board(n_givens)   # create an empty sudoku board with n given numbers

# print(state[0])

# state = smart_fill(state)

# E = energy(state)
# t = 0

# E_history = [E] 
# T_history = [T_max]

### RUNNING THE ALGORITHM ###

# progress = [n * t_max / 10 for n in range(1, 11)]

# while E > 0 and t < t_max:
#     T = exp_T(t, t_max, T_max=T_max, n_cycles=10, frac=1)
    
#     if t in progress:      # progress updater
#         print(f"PROGRESS = {100 * t / t_max}%")

#     state_new = change_t4(state)
#     E_new = energy(state_new)

#     if choose_metropolis(E, E_new, T):
#         state = state_new
#         E = E_new

#     E_history.append(E)
#     T_history.append(T)
#     t += 1
# print("PROGRESS = 100.0%")

# if E == 0:
#     print(f"SOLVED: E = {E}")
#     print(state[0])

# plot_E_T(E_history, T_history, t)


rand_sudoku = SudokuBoard(*set_given_board(17))
print(rand_sudoku)
rand_sudoku.anneal_solve()
print(rand_sudoku)