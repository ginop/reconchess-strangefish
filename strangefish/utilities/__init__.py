from functools import partial
from typing import Iterable, Optional, Set
from collections import defaultdict
from tqdm import tqdm
import signal

import chess

from reconchess.utilities import without_opponent_pieces, is_illegal_castle, is_psuedo_legal_castle, slide_move, \
    moves_without_opponent_pieces, pawn_capture_moves_on, capture_square_of_move

# These are the possible squares to search–all squares that aren't on the edge of the board.
SEARCH_SPOTS = [
    9, 10, 11, 12, 13, 14,
    17, 18, 19, 20, 21, 22,
    25, 26, 27, 28, 29, 30,
    33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46,
    49, 50, 51, 52, 53, 54,
]
SEARCH_OFFSETS = [-9, -8, -7, -1, 0, 1, 7, 8, 9]


# Generate all RBC-legal moves for a board
def generate_rbc_moves(board: chess.Board) -> Iterable[chess.Move]:
    for move in board.pseudo_legal_moves:
        yield move
    for move in without_opponent_pieces(board).generate_castling_moves():
        if not is_illegal_castle(board, move):
            yield move
    yield chess.Move.null()


# Generate all possible moves from just our own pieces
def generate_moves_without_opponent_pieces(board: chess.Board) -> Iterable[chess.Move]:
    for move in moves_without_opponent_pieces(board):
        yield move
    for move in pawn_capture_moves_on(board):
        yield move
    yield chess.Move.null()


# Produce a sense result from a hypothetical true board and a sense square
def simulate_sense(board, square):  # copied (with modifications) from LocalGame
    if square is None:
        # don't sense anything
        sense_result = []
    else:
        if square not in list(chess.SQUARES):
            raise ValueError('LocalGame::sense({}): {} is not a valid square.'.format(square, square))
        rank, file = chess.square_rank(square), chess.square_file(square)
        sense_result = []
        for delta_rank in [1, 0, -1]:
            for delta_file in [-1, 0, 1]:
                if 0 <= rank + delta_rank <= 7 and 0 <= file + delta_file <= 7:
                    sense_square = chess.square(file + delta_file, rank + delta_rank)
                    sense_result.append((sense_square, board.piece_at(sense_square)))
    return tuple(sense_result)


# test an attempted move on a board to see what move is actually taken
def simulate_move(board, move):
    if move == chess.Move.null():
        return None
    # if its a legal move, don't change it at all (generate_pseudo_legal_moves does not include pseudo legal castles)
    if move in board.generate_pseudo_legal_moves() or is_psuedo_legal_castle(board, move):
        return move
    if is_illegal_castle(board, move):
        return None
    # if the piece is a sliding piece, slide it as far as it can go
    piece = board.piece_at(move.from_square)
    if piece.piece_type in [chess.PAWN, chess.ROOK, chess.BISHOP, chess.QUEEN]:
        move = slide_move(board, move)
    return move if move in board.generate_pseudo_legal_moves() else None


# check if a taken move would have happened on a board
def validate_move_on_board(epd, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[chess.Square]) -> bool:
    board = chess.Board(epd)
    # if the taken move was a capture...
    if captured_opponent_piece:
        # the board is invalid if the capture would not have happened
        if not board.is_capture(taken_move):
            return False
        # the board is invalid if the captured piece would have been the king
        # (wrong if it really was the king, but then the game is over)
        captured_piece = board.piece_at(capture_square)
        if captured_piece and captured_piece.piece_type == chess.KING:
            return False
    # if the taken move was not a capture...
    elif taken_move != chess.Move.null():
        # the board is invalid if a capture would have happened
        if board.is_capture(taken_move):
            return False
    # invalid if the requested move would have not resulted in the taken move
    if (simulate_move(board, requested_move) or chess.Move.null()) != taken_move:
        return False
    # otherwise the board is still valid
    return True


# Expand one turn's boards into next turn's set by all possible moves. Store as dictionary keyed by capture square.
def populate_next_board_set(board_set: Set[str], my_color, pool=None, rc_disable_pbar: bool = False):
    next_turn_boards = defaultdict(set)
    iter_boards = tqdm(board_set, disable=rc_disable_pbar, unit='boards',
                       desc=f'{chess.COLOR_NAMES[my_color]} Expanding {len(board_set)} boards into new set')
    all_pairs = (pool.imap_unordered(partial(get_next_boards_and_capture_squares, my_color), iter_boards)
                 if pool else map(partial(get_next_boards_and_capture_squares, my_color), iter_boards))
    for pairs in all_pairs:
        for capture_square, next_epd in pairs:
            next_turn_boards[capture_square].add(next_epd)
    return next_turn_boards


# Check if a board could have produced these sense results
def board_matches_sense(board_epd, sense_result):
    board = chess.Board(board_epd)
    for square, piece in sense_result:
        if board.piece_at(square) != piece:
            return None
    return board_epd


# Check if a requested move - taken move pair would have been produced on this board
def move_would_happen_on_board(requested_move, taken_move, captured_opponent_piece, capture_square, board_epd):
    if validate_move_on_board(board_epd, requested_move, taken_move, captured_opponent_piece, capture_square):
        return push_move_to_epd(board_epd, taken_move)


# Change an EPD string to reflect a move
def push_move_to_epd(epd, move):
    board = chess.Board(epd)
    board.push(move)
    return board.epd(en_passant='xfen')


# Generate tuples of next turn's boards and capture squares for one current board
def get_next_boards_and_capture_squares(my_color, board_epd):
    board = chess.Board(board_epd)
    # Calculate all possible opponent moves from this board state
    board.turn = not my_color
    pairs = []
    for move in generate_rbc_moves(board):
        next_board = board.copy()
        next_board.push(move)
        capture_square = capture_square_of_move(board, move)
        next_epd = next_board.epd(en_passant='xfen')
        pairs.append((capture_square, next_epd))
    return pairs


# Change any promotion moves to choose queen
def force_promotion_to_queen(move: chess.Move):
    return move if len(move.uci()) == 4 else chess.Move.from_uci(move.uci()[:4] + 'q')


def ignore_one_term(signum, frame):  # Let a sub-process survive the first ctrl-c call for graceful game exiting
    # reset to default response to interrupt signals
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
