import multiprocessing
import time
import traceback
import yaml
import click
import requests

import chess
from reconchess import Player, RemoteGame, play_turn, ChessJSONDecoder
from reconchess.scripts.rc_connect import RBCServer

from strangefish import StrangeFish
from strangefish.strategies import multiprocessing_strategies
from strangefish.strategies.multiprocessing_strategies import MoveConfig, TimeConfig, ScoreConfig
from strangefish.utilities.player_logging import create_main_logger, create_sub_logger


class OurRemoteGame(RemoteGame):

    def __init__(self, *args, **kwargs):
        self.logger = create_sub_logger('game_communications')
        super().__init__(*args, **kwargs)

    def is_op_turn(self):
        status = self._get('game_status')
        return not status['is_over'] and not status['is_my_turn']

    def _get(self, endpoint, decoder_cls=ChessJSONDecoder):
        self.logger.debug(f"Getting '{endpoint}'")
        return super()._get(endpoint, decoder_cls)

    def _post(self, endpoint, obj):
        self.logger.debug(f"Posting '{endpoint}' -> {obj}")
        return super()._post(endpoint, obj)


def our_play_remote_game(server_url, game_id, auth, player: Player):
    game = OurRemoteGame(server_url, game_id, auth)
    logger = create_sub_logger('game_moderator')

    op_name = game.get_opponent_name()
    our_color = game.get_player_color()

    logger.debug("Setting up remote game %d playing %s against %s.",
                 game_id, chess.COLOR_NAMES[our_color], op_name)

    player.handle_game_start(our_color, game.get_starting_board(), op_name)
    game.start()

    turn_num = 0
    while not game.is_over():
        turn_num += 1
        logger.info("Playing turn %2d. (%3.0f seconds left.)", turn_num, game.get_seconds_left())
        play_turn(game, player, end_turn_last=False)
        logger.info("   Done turn %2d.", turn_num)

        if hasattr(player, 'while_we_wait') and getattr(player, 'while_we_wait'):
            while game.is_op_turn():
                player.while_we_wait()

    winner_color = game.get_winner_color()
    win_reason = game.get_win_reason()
    game_history = game.get_game_history()

    logger.debug("Ending remote game %d against %s.", game_id, op_name)

    player.handle_game_end(winner_color, win_reason, game_history)

    return winner_color, win_reason, game_history


def accept_invitation_and_play(server_url, auth, invitation_id):
    player = get_player_from_config()
    logger = create_sub_logger('invitations')

    logger.debug('Accepting invitation %d.', invitation_id)
    server = RBCServer(server_url, auth)
    game_id = server.accept_invitation(invitation_id)
    logger.info('Invitation %d accepted. Playing game %d.', invitation_id, game_id)

    try:
        our_play_remote_game(server_url, game_id, auth, player)
        logger.debug('Finished game %d.', game_id)
    except:
        logger.error('Fatal error in game %d.', game_id)
        traceback.print_exc()
        server.error_resign(game_id)
        player.handle_game_end(None, None, None)
        logger.critical('Game %d closed on account of error.', game_id)

    server.finish_invitation(invitation_id)
    logger.debug('Game %d ended. Invitation %d closed.', game_id, invitation_id)


def listen_for_invitations(server_url, auth, max_concurrent_games):
    server = RBCServer(server_url, auth)

    create_main_logger(log_to_file=True)
    logger = create_sub_logger('server_manager')

    connected = False
    queued_invitations = set()
    current_games = []

    while True:
        try:
            invitations = server.get_invitations()

            if not connected:
                logger.info('Connected successfully to server!')
                connected = True
                server.set_max_games(max_concurrent_games)

            unqueued_invitations = set(invitations) - queued_invitations
            for invitation_id in unqueued_invitations:
                if len(current_games) < max_concurrent_games:
                    logger.debug(f'Received invitation {invitation_id}.')
                    queued_invitations.add(invitation_id)
                    current_games.append(multiprocessing.Process(
                        target=accept_invitation_and_play,
                        args=(server_url, auth, invitation_id)
                        ))
                    current_games[-1].start()

            for game in current_games:
                if not game.is_alive():
                    current_games.remove(game)

        except requests.RequestException as e:
            connected = False
            logger.exception('Failed to connect to server')
        except Exception:
            logger.exception("Error in invitation processing: ")

        time.sleep(5)


def get_player_from_config():
    create_main_logger(log_to_file=True)
    logger = create_sub_logger('config_loading')
    file_loaded = False
    while not file_loaded:
        logger.debug("Loading config.yaml for player settings.")
        try:
            with open('config.yml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            player = player_configuration(config)
            file_loaded = True
        except Exception:
            logger.exception("Something went wrong loading config.yaml. Attempting to load backup_config.yaml next.")
            try:
                with open('backup_config.yml') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                player = player_configuration(config)
                file_loaded = True
            except Exception:
                logger.exception("Also failed to load backup_config.yaml. Repeating loading loop.")
                time.sleep(0.1)

    return player


def player_configuration(config):
    logger = create_sub_logger('config_loading')

    strategy_config = config.pop('multiprocessing_strategies')
    score_config = ScoreConfig(**strategy_config.pop('score_config'))
    move_config = MoveConfig(**strategy_config.pop('move_config'))
    time_config = TimeConfig(**strategy_config.pop('time_config'))
    strategy = multiprocessing_strategies.create_strategy(
        **strategy_config,
        score_config=score_config,
        move_config=move_config,
        time_config=time_config,
    )
    strangefish = StrangeFish(*strategy, **config)

    logger.debug(
        "Created a StrangeFish player using multiprocessing_strategies with the following configuration: "
        f"score_config = {score_config}, "
        f"move_config = {move_config}, "
        f"time_config = {time_config}, "
        f"other strategy arguments = {strategy_config}, "
        f"other player arguments = {config}, "
    )

    return strangefish


@click.command()
@click.argument('username')
@click.argument('password')
@click.option('--server-url', 'server_url', default='https://rbc.jhuapl.edu', help='URL of the server.')
@click.option('--max-concurrent-games', 'max_concurrent_games', type=int, default=4, help='Maximum games to play at once.')
def main(username, password, server_url, max_concurrent_games):
    listen_for_invitations(server_url, (username, password), max_concurrent_games)


if __name__ == '__main__':
    main()
