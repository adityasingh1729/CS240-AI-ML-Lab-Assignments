import copy
import math
import logging
import sys

sys.setrecursionlimit(10**6)

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

board_positions_val_dict = {}
visited_histories_list = []


class History:
    def __init__(self, num_boards=2, history=None):
        self.num_boards = num_boards
        if history is not None:
            self.history = history
            self.boards = self.get_boards()
        else:
            self.history = []
            self.boards = []
            for _ in range(self.num_boards):
                self.boards.append(['0', '0', '0', '0', '0', '0', '0', '0', '0'])
        self.active_board_stats = self.check_active_boards()
        self.current_player = self.get_current_player()

    def get_boards(self):
        boards = []
        for _ in range(self.num_boards):
            boards.append(['0', '0', '0', '0', '0', '0', '0', '0', '0'])
        for move in self.history:
            board_num = math.floor(move / 9)
            play_position = move % 9
            boards[board_num][play_position] = 'x'
        return boards

    def check_active_boards(self):
        active_board_stat = []
        for i in range(self.num_boards):
            if self.is_board_win(self.boards[i]):
                active_board_stat.append(0)
            else:
                active_board_stat.append(1)
        return active_board_stat

    @staticmethod
    def is_board_win(board):
        for i in range(3):
            if board[3 * i] == board[3 * i + 1] == board[3 * i + 2] != '0':
                return True

            if board[i] == board[i + 3] == board[i + 6] != '0':
                return True

        if board[0] == board[4] == board[8] != '0':
            return True

        if board[2] == board[4] == board[6] != '0':
            return True
        return False

    def get_current_player(self):
        total_num_moves = len(self.history)
        if total_num_moves % 2 == 0:
            return 1
        else:
            return 2

    def get_boards_str(self):
        boards_str = ""
        for i in range(self.num_boards):
            boards_str = boards_str + ''.join([str(j) for j in self.boards[i]])
        return boards_str

    def is_terminal_history(self):
        for board in self.boards:
            if self.is_board_win(board):
                return True
        return False

    def get_valid_actions(self):
        valid_actions = []
        for i, board in enumerate(self.boards):
            if self.active_board_stats[i] == 1:
                for j, square in enumerate(board):
                    if square == '0':
                        valid_actions.append(i * 9 + j)
        return valid_actions

    def get_value_given_terminal_history(self):
        if self.is_terminal_history():
            return -1  # Assuming player 1 loses in a terminal history
        else:
            return 0  # Draw or ongoing game


def alpha_beta_pruning(history_obj, alpha, beta, max_player_flag, depth, memo):
    global visited_histories_list
    visited_histories_list.append(history_obj.history)

    if history_obj.is_terminal_history() or depth == 0:
        return history_obj.get_value_given_terminal_history()

    boards_str = history_obj.get_boards_str()
    if boards_str in memo:
        return memo[boards_str]

    valid_actions = history_obj.get_valid_actions()
    ordered_actions = [4, 0, 2, 6, 8, 1, 3, 5, 7]  # Order actions: center, corners, remaining

    if max_player_flag:
        max_val = -math.inf
        for action in ordered_actions:
            if action in valid_actions:
                new_history = copy.deepcopy(history_obj)
                new_history.history.append(action)
                val = alpha_beta_pruning(new_history, alpha, beta, False, depth - 1, memo)
                max_val = max(max_val, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break  # Beta cut-off
        memo[boards_str] = max_val
        return max_val
    else:
        min_val = math.inf
        for action in ordered_actions:
            if action in valid_actions:
                new_history = copy.deepcopy(history_obj)
                new_history.history.append(action)
                val = alpha_beta_pruning(new_history, alpha, beta, True, depth - 1, memo)
                min_val = min(min_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break  # Alpha cut-off
        memo[boards_str] = min_val
        return min_val


def maxmin(history_obj, max_player_flag):
    global board_positions_val_dict

    if history_obj.is_terminal_history():
        return history_obj.get_value_given_terminal_history()

    boards_str = history_obj.get_boards_str()
    if boards_str in board_positions_val_dict:
        return board_positions_val_dict[boards_str]

    valid_actions = history_obj.get_valid_actions()
    max_val = -math.inf if max_player_flag else math.inf

    for action in valid_actions:
        new_history = copy.deepcopy(history_obj)
        new_history.history.append(action)
        val = maxmin(new_history, not max_player_flag)
        if max_player_flag:
            max_val = max(max_val, val)
        else:
            max_val = min(max_val, val)

    board_positions_val_dict[boards_str] = max_val
    return max_val


def solve_alpha_beta_pruning(history_obj, alpha, beta, max_player_flag):
    global visited_histories_list
    memo = {}  # Initialize memoization dictionary
    val = alpha_beta_pruning(history_obj, alpha, beta, max_player_flag, float('inf'), memo)
    return val, visited_histories_list


if __name__ == "__main__":
    logging.info("start")
    logging.info("alpha beta pruning")
    value, visited_histories = solve_alpha_beta_pruning(History(history=[], num_boards=2), -math.inf, math.inf, True)
    logging.info("maxmin value {}".format(value))
    logging.info("Number of histories visited {}".format(len(visited_histories)))
    logging.info("maxmin memory")
    logging.info("maxmin value {}".format(maxmin(History(history=[], num_boards=2), True)))
    logging.info("end")
