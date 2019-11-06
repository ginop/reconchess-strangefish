import multiprocessing as mp
import os
from collections import defaultdict
from functools import partial
from time import time
from tqdm import tqdm
from typing import Optional, List, Tuple, Set, Callable

import chess.engine
from reconchess import Player, Color, GameHistory, WinReason, Square

from strangefish import defaults
from strangefish.strategies.auxiliary_strategies import contingency_strategy
from strangefish.utilities import board_matches_sense, move_would_happen_on_board, populate_next_board_set
from strangefish.utilities.player_logging import create_main_logger
from strangefish.utilities.timing import Timer

# Parameters for minor bot behaviors
RC_DISABLE_PBAR = os.getenv('RC_DISABLE_PBAR', 'false').lower() == 'true'  # Flag to disable the tqdm progress bars
WAIT_LOOP_RATE_LIMIT = 1  # minimum seconds spent looping in self.while_we_wait()

# Parameters for switching to the emergency backup plan
BOARD_SET_LIMIT = 1_000_000  # number of boards in set at which we stop processing and revert to backup plan
TIMEOUT_RESIGN_LIMIT = 10  # number of seconds left at which we stop processing and revert to backup plan
AVG_BOARD_EXP = 33  # number of moves on each board: mean 33, std 10 according to ~320k boards in logs


class StrangeFish(Player):
    """
    StrangeFish is the main skeleton of our reconchess-playing bot. Its primary role is to manage the set of all
    possible board states based on the given information. Decision making for sense and move choices are handed off to
    one of our strategy functions.

    StrangeFish alone does not use the Stockfish chess engine, but most of our strategies do use it to make sensing and
    moving decisions. In order to run StrangeFish with one of those strategies, you'll need to download Stockfish from
    https://stockfishchess.org/download/ and create an environment variable called STOCKFISH_EXECUTABLE that is the path
    to the downloaded Stockfish executable.
    """

    def __init__(
        self,

        choose_sense: Callable[[Set[str], bool, List[Square], List[chess.Move], float], Square] = defaults.choose_sense,
        choose_move: Callable[[Set[str], bool, List[chess.Move], float], chess.Move] = defaults.choose_move,
        while_we_wait: Optional[Callable[[Set[str], bool], None]] = defaults.while_we_wait,
        end_game: Optional[Callable[[Set[str]], None]] = defaults.end_game,

        pool_size: Optional[int] = 2,
        log_to_file=True,
        save_debug_history=False,
        rc_disable_pbar=RC_DISABLE_PBAR,
    ):
        """
        Set up StrangeFish with decision-making capabilities inherited from another function.

        :param choose_sense: A callable produced by the strategy function which chooses and returns the sense square
        :param choose_move: A callable produced by the strategy function which chooses and returns the move
        :param while_we_wait: An optional callable produced by the strategy function which uses time between our turns
        :param end_game: An optional callable produced by the strategy function which (typically) shuts down StockFish

        :param pool_size: Number of processes to use when multiprocessing board set expansion and filtering
        :param log_to_file: A boolean flag to turn on/off logging to file gameLogs/StrangeFish.log
        :param save_debug_history: A boolean flag to turn on/off the generation of a turn-by-turn internal history
        :param rc_disable_pbar: A boolean flag to turn on/off the tqdm progress bars
        """

        self._choose_sense = choose_sense
        self._choose_move = choose_move
        self._while_we_wait = while_we_wait
        self._end_game = end_game

        self.boards: Set[str] = set()
        self.next_turn_boards: defaultdict[Set] = defaultdict(set)
        self.next_turn_boards_unsorted: Set[str] = set()

        self.color = None
        self.turn_num = None

        self.pool = mp.Pool(pool_size)

        self.save_debug_history = save_debug_history
        self.debug_memory = []

        self.rc_disable_pbar = rc_disable_pbar

        self.timeout_resign = None  # flag used to skip later turn processes if we have run out of time

        self.logger = create_main_logger(log_to_file=log_to_file)
        self.logger.debug("A new StrangeFish player was initialized.")

    def _game_state_log(self, step_name='-'):  # Save game state for advanced replay
        if self.save_debug_history:
            info = {
                'name': __name__,
                'color': chess.COLOR_NAMES[self.color],
                'turn': self.turn_num,
                'step': step_name,
                'boards': list(self.boards),
            }
            self.debug_memory.append(info)

    def _emergency_plan(self):  # Switch to emergency backup plan
        self.boards = set()
        self.next_turn_boards = {None: set()}
        self._choose_sense, self._choose_move = contingency_strategy()
        setattr(self, 'while_we_wait', None)

    def get_debug_history(self):  # Get possible board states from each turn
        return self.debug_memory

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        color_name = chess.COLOR_NAMES[color]

        self.logger.info('Starting a new game as %s against %s.', color_name, opponent_name)
        self.boards = {board.epd(en_passant='xfen')}
        self.color = color
        self.turn_num = 0
        self.timeout_resign = False

        # Save game state for advanced replay
        if self.color == chess.BLACK:
            self._game_state_log()
            self._game_state_log()

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        self.turn_num += 1
        self.logger.debug("Starting turn %d.", self.turn_num)

        # Do not "handle_opponent_move_result" if no one has moved yet
        if self.turn_num == 1 and self.color == chess.WHITE:
            self._game_state_log()
            return

        if captured_my_piece:
            self.logger.debug('Opponent captured my piece at %s.', chess.SQUARE_NAMES[capture_square])
        else:
            self.logger.debug("Opponent's move was not a capture.")

        self.logger.debug('Already calculated scores for %d possible boards, '
                          'approximately %d x %d = %d boards left to analyse.',
                          len(self.next_turn_boards[None]), len(self.boards),
                          AVG_BOARD_EXP, (AVG_BOARD_EXP * len(self.boards)))

        # Check for board set over-growth and switch to emergency plan if needed
        if not captured_my_piece and \
                (len(self.next_turn_boards[None]) + (AVG_BOARD_EXP * len(self.boards))) > BOARD_SET_LIMIT:
            self.logger.warning("Board set grew too large, switching to contingency plan. "
                                "Set size expected to grow to %d; limit is %d",
                                len(self.next_turn_boards[None]) + (AVG_BOARD_EXP * len(self.boards)),
                                BOARD_SET_LIMIT)
            self._emergency_plan()

        # If creation of new board set didn't complete during op's turn (self.boards will not be empty)
        if self.boards:
            new_board_set = populate_next_board_set(self.boards, self.color, self.pool,
                                                    rc_disable_pbar=self.rc_disable_pbar)
            for square in new_board_set.keys():
                self.next_turn_boards[square] |= new_board_set[square]

        # Get this turn's board set from a dictionary keyed by the possible capture squares
        self.boards = self.next_turn_boards[capture_square]

        self.logger.debug('Finished expanding and filtering the set of possible board states. '
                          'There are %d possible boards at the start of our turn %d.',
                          len(self.boards), self.turn_num)

        # Save game state for advanced replay
        self._game_state_log('post-op-move')

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float
                     ) -> Optional[Square]:
        # Check if time is up (or if we already changed to the emergency plan)
        if not self.timeout_resign and seconds_left <= TIMEOUT_RESIGN_LIMIT:
            self.logger.warning(f'Time is nearly up, go to backup plan.')
            self._emergency_plan()
            self.timeout_resign = True

        self.logger.debug('Choosing a sensing square for turn %d with %d boards and %.0f seconds remaining.',
                          self.turn_num, len(self.boards), seconds_left)

        # The option to pass isn't included in the reconchess input
        move_actions += [chess.Move.null()]

        with Timer(self.logger.debug, 'choosing sense location'):
            # Pass the needed information to the decision-making function to choose a sense square
            sense_choice = self._choose_sense(self.boards, self.color, sense_actions, move_actions, seconds_left)

        self.logger.debug('Chose to sense %s', chess.SQUARE_NAMES[sense_choice] if sense_choice else 'nowhere')

        return sense_choice

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):

        # Filter the possible board set to only boards which would have produced the observed sense result
        num_before = len(self.boards)
        i = tqdm(
            self.boards,
            disable=self.rc_disable_pbar,
            desc=f'{chess.COLOR_NAMES[self.color]} Filtering {len(self.boards)} boards by sense results',
            unit='boards',
        )
        self.boards = {board_epd for board_epd in
                       self.pool.imap_unordered(partial(board_matches_sense, sense_result=sense_result), i)
                       if board_epd is not None}
        self.logger.debug('There were %d possible boards before sensing and %d after.', num_before, len(self.boards))

        # Save game state for advanced replay
        self._game_state_log('post-sense')

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:

        # Currently, move_actions is passed by reference, so if we add the null move here it will be in the list twice
        #  since we added it in choose_sense also. Instead of removing this line altogether, I'm leaving a check so we
        #  are prepared in the case that reconchess is updated to pass a copy of the move_actions list instead.
        if chess.Move.null() not in move_actions:
            move_actions += [chess.Move.null()]

        self.logger.debug('Choosing move for turn %d from %d moves over %d boards with %.2f seconds remaining.',
                          self.turn_num, len(move_actions), len(self.boards), seconds_left)

        with Timer(self.logger.debug, 'choosing move'):
            # Pass the needed information to the decision-making function to choose a move
            move_choice = self._choose_move(self.boards, self.color, move_actions, seconds_left)

        self.logger.debug('The chosen move was %s', move_choice)

        # reconchess uses None for the null move, so correct the function output if that was our choice
        return move_choice if move_choice != chess.Move.null() else None

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):

        self.logger.debug('The requested move was %s and the taken move was %s.', requested_move, taken_move)
        if captured_opponent_piece:
            self.logger.debug('Move %s was a capture!', taken_move)

        num_boards_before_filtering = len(self.boards)

        if requested_move is None:
            requested_move = chess.Move.null()
        if taken_move is None:
            taken_move = chess.Move.null()

        # Filter the possible board set to only boards on which the requested move would have resulted in the taken move
        i = tqdm(
            self.boards,
            disable=self.rc_disable_pbar,
            desc=f'{chess.COLOR_NAMES[self.color]} Filtering {len(self.boards)} boards by move results',
            unit='boards',
        )
        self.boards = {
            board_epd for board_epd in
            self.pool.imap_unordered(partial(move_would_happen_on_board, requested_move, taken_move,
                                             captured_opponent_piece, capture_square), i)
            if board_epd is not None
        }

        self.logger.debug('There were %d possible boards before filtering and %d after.',
                          num_boards_before_filtering, len(self.boards))

        # Save game state for advanced replay
        self._game_state_log('post-move')
        self._game_state_log()

        # Re-initialize the set of boards for next turn (filled in while_we_wait and/or handle_opponent_move_result)
        self.next_turn_boards = defaultdict(set)
        self.next_turn_boards_unsorted = set()

    def while_we_wait(self):
        start_time = time()
        self.logger.debug('Running the "while_we_wait" method. '
                          f'{len(self.boards)} boards left to expand for next turn.')

        our_king_square = chess.Board(tuple(self.boards)[0]).king(self.color) if len(self.boards) else None

        while time() - start_time < WAIT_LOOP_RATE_LIMIT:

            # If there are still boards in the set from last turn, remove one and expand it by all possible moves
            if len(self.boards):
                new_board_set = populate_next_board_set({self.boards.pop()}, self.color, rc_disable_pbar=True)
                for square in new_board_set.keys():
                    self.next_turn_boards[square] |= new_board_set[square]
                    if square != our_king_square:
                        self.next_turn_boards_unsorted |= new_board_set[square]

            # If all of last turn's boards have been expanded, pass to the sense/move function's waiting method
            elif self._while_we_wait:
                self._while_we_wait(self.next_turn_boards_unsorted, self.color)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory
                        ):
        self.logger.info('I %s by %s', "won" if winner_color == self.color else "lost",
                         win_reason.name if hasattr(win_reason, "name") else win_reason)
        self.pool.terminate()
        self._end_game()
