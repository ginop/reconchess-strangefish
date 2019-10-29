import json
from typing import Set
from tqdm import tqdm

import chess.engine
from reconchess.utilities import moves_without_opponent_pieces, pawn_capture_moves_on

from strangefish.utilities import stockfish, populate_next_board_set, simulate_move
from strangefish.utilities.rbc_move_score import calculate_score, ScoreConfig


score_config: ScoreConfig = ScoreConfig(search_depth=16)
engine = stockfish.create_engine()

score_cache = dict()
boards_in_cache = set()


# Create a cache key for the requested board and move (keyed based on the move that would result from that request)
def make_cache_key(board: chess.Board, move: chess.Move = chess.Move.null(), prev_turn_score: int = None):
    move = simulate_move(board, move) or chess.Move.null()
    return (board.epd(en_passant="xfen") + ' ' + move.uci() + ' ' +
            (str(prev_turn_score) if prev_turn_score is not None else '-'))


# Memoized calculation of the score associated with one move on one board
def memo_calc_score(board: chess.Board, move: chess.Move = chess.Move.null(), prev_turn_score=None):
    key = make_cache_key(board, move, prev_turn_score)
    if key in score_cache:
        return score_cache[key]
    result = calculate_score(board=board, move=move, prev_turn_score=prev_turn_score or 0,
                             score_config=score_config, engine=engine)
    score_cache[key] = result
    return result


def main():
    # Initialize the board set as the opening position
    board_set: Set[str] = set()
    board_set.add(chess.Board().epd(en_passant="xfen"))

    # Find all boards that can be reached in num_half_turns moves
    num_half_turns = 1
    color_to_play = chess.WHITE
    for i in range(num_half_turns):
        next_turn_boards = populate_next_board_set(board_set, not color_to_play)
        board_set |= set().union(*next_turn_boards.values())
        color_to_play = not color_to_play

    # Calculate and cache scores for all moves on all boards in the set
    for board_epd in tqdm(board_set, desc=f'Scoring moves from {len(board_set)} early-game board states', unit='boards'):
        board = chess.Board(board_epd)
        if board.king(chess.WHITE) and board.king(chess.BLACK):
            board.turn = not board.turn
            score = memo_calc_score(board=board)
            board.turn = not board.turn
            boards_in_cache.add(board.epd(en_passant="xfen"))
            for move in moves_without_opponent_pieces(board) + pawn_capture_moves_on(board) + [chess.Move.null()]:
                memo_calc_score(board=board, move=move, prev_turn_score=-score)

    # Store the cache as a json file
    with open('strangefish/score_cache.json', 'w') as file:
        json.dump({
            'cache': score_cache,
            'boards': list(boards_in_cache)
        }, file)

    # Shut down Stockfish
    try:
        engine.quit()
    except chess.engine.EngineTerminatedError:
        pass


if __name__ == '__main__':
    main()
