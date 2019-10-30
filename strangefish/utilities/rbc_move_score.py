from dataclasses import dataclass

import chess.engine
from reconchess import is_psuedo_legal_castle
from reconchess.utilities import capture_square_of_move

from strangefish.utilities import simulate_move


@dataclass
class ScoreConfig:
    capture_king_score: float = 50_000  # bonus points for a winning move
    checkmate_score: int = 30_000  # point value of checkmate
    into_check_score: float = -40_000  # point penalty for moving into check
    search_depth: int = 8  # Stockfish engine search ply
    reward_attacker: float = 300  # Bonus points if move sets up attack on enemy king
    require_sneak: bool = True  # Only reward bonus points to aggressive moves if they are sneaky (aren't captures)


def calculate_score(engine: chess.engine.SimpleEngine,
                    board, move=chess.Move.null(),
                    prev_turn_score=0,
                    score_config: ScoreConfig = ScoreConfig()):

    pov = board.turn

    if move != chess.Move.null() and not is_psuedo_legal_castle(board, move):

        if not board.is_pseudo_legal(move):
            # check for sliding move alternate results, and score accordingly
            revised_move = simulate_move(board, move)
            if revised_move is not None:
                return calculate_score(engine, board, revised_move, prev_turn_score, score_config)
            return calculate_score(engine, board, chess.Move.null(), prev_turn_score, score_config)

        if board.is_capture(move):
            if board.piece_at(capture_square_of_move(board, move)).piece_type is chess.KING:
                return score_config.capture_king_score

    next_board = board.copy()
    next_board.push(move)
    next_board.clear_stack()

    if next_board.was_into_check():
        return score_config.into_check_score

    engine_result = engine.analyse(next_board, chess.engine.Limit(depth=score_config.search_depth))
    score = engine_result.score.pov(pov).score(mate_score=score_config.checkmate_score)

    # Add bonus board position score if king is attacked
    king_attackers = next_board.attackers(pov, next_board.king(not pov))  # list of pieces that can reach the enemy king
    if king_attackers:  # if there are any such pieces...
        if not score_config.require_sneak:  # and we don't require the attackers to be sneaky
            score += score_config.reward_attacker  # add the bonus points
        # or if we do require the attackers to be sneaky, either the last move was not a capture (which would give away
        # our position) or there are now attackers other than the piece that moves (discovered check)
        elif not next_board.is_capture(move) or any([square != move.to_square for square in king_attackers]):
            score += score_config.reward_attacker  # add the bonus points

    score -= prev_turn_score

    return score
