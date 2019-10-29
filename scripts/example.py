import traceback
from datetime import datetime

import chess
from reconchess import LocalGame, play_local_game
from reconchess.bots.trout_bot import TroutBot

from strangefish import StrangeFish
from strangefish.strategies import multiprocessing_strategies


def main():
    white_bot_name, black_bot_name = 'TroutBot', 'StrangeFish'

    game = LocalGame()

    try:
        winner_color, win_reason, history = play_local_game(
            TroutBot(),
            StrangeFish(*multiprocessing_strategies.create_strategy()),
            game=game
        )

        winner = 'Draw' if winner_color is None else chess.COLOR_NAMES[winner_color]
    except:
        traceback.print_exc()
        game.end()

        winner = 'ERROR'
        history = game.get_game_history()

    print('Game Over!')
    print('Winner: {}!'.format(winner))

    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    replay_path = '{}-{}-{}-{}.json'.format(white_bot_name, black_bot_name, winner, timestamp)
    print('Saving replay to {}...'.format(replay_path))
    history.save(replay_path)


if __name__ == '__main__':
    main()
