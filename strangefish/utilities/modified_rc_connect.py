import multiprocessing
import signal
import sys
import time
import traceback
import yaml
import click
import requests

import chess
from reconchess import Player, RemoteGame, play_turn, ChessJSONDecoder
from reconchess.scripts.rc_connect import RBCServer, ranked_mode, unranked_mode, check_package_version

from strangefish import StrangeFish
from strangefish.strategies import multiprocessing_strategies
from strangefish.strategies.multiprocessing_strategies import MoveConfig, TimeConfig, ScoreConfig
from strangefish.utilities import ignore_one_term
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


def accept_invitation_and_play(server_url, auth, invitation_id, finished):
    # make sure this process doesn't react to the first interrupt signal
    signal.signal(signal.SIGINT, ignore_one_term)

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
    finally:
        server.finish_invitation(invitation_id)
        finished.value = True
        logger.debug('Game %d ended. Invitation %d closed.', game_id, invitation_id)


def listen_for_invitations(server, max_concurrent_games):

    logger = create_sub_logger('server_manager')

    connected = False
    process_by_invitation = {}
    finished_by_invitation = {}
    while True:
        try:
            # get unaccepted invitations
            invitations = server.get_invitations()

            # set max games on server if this is the first successful connection after being disconnected
            if not connected:
                logger.info('Connected successfully to server!')
                connected = True
                server.set_max_games(max_concurrent_games)

            # filter out finished processes
            finished_invitations = []
            for invitation in process_by_invitation.keys():
                if not process_by_invitation[invitation].is_alive() or finished_by_invitation[invitation].value:
                    finished_invitations.append(invitation)
            for invitation in finished_invitations:
                logger.info(f'Terminating process for invitation {invitation}')
                process_by_invitation[invitation].terminate()
                del process_by_invitation[invitation]
                del finished_by_invitation[invitation]

            # accept invitations until we have #max_concurrent_games processes alive
            for invitation in invitations:
                # only accept the invitation if we have room and the invite doesn't have a process already
                if invitation not in process_by_invitation:
                    logger.debug(f'Received invitation {invitation}.')

                    if len(process_by_invitation) < max_concurrent_games:
                        # start the process for playing a game
                        finished = multiprocessing.Value('b', False)
                        process = multiprocessing.Process(
                            target=accept_invitation_and_play,
                            args=(server.server_url, server.session.auth, invitation, finished))
                        process.start()

                        # store the process so we can check when it finishes
                        process_by_invitation[invitation] = process
                        finished_by_invitation[invitation] = finished
                    else:
                        logger.info(f'Not enough game slots to play invitation {invitation}.')
                        unranked_mode(server)
                        max_concurrent_games += 1

        except requests.RequestException as e:
            connected = False
            logger.exception('Failed to connect to server')
            print(e)
        except Exception:
            logger.exception("Error in invitation processing: ")
            traceback.print_exc()

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
@click.option('--max-concurrent-games', 'max_concurrent_games', type=int, default=1, help='Maximum games to play at once.')
@click.option('--ranked', 'ranked', type=bool, default=False, help='Play for leaderboard ELO.')
@click.option('--keep-version', 'keep_version', type=bool, default=True, help='Keep existing leaderboard version num.')
def main(username, password, server_url, max_concurrent_games, ranked, keep_version):
    create_main_logger(log_to_file=True)
    logger = create_sub_logger('modified_rc_connect')

    auth = username, password
    server = RBCServer(server_url, auth)

    # verify we have the correct version of reconchess package
    check_package_version(server)

    def handle_term(signum, frame):
        # reset to default response to interrupt signals
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        logger.warning('Received terminate signal, waiting for games to finish and then exiting.')
        unranked_mode(server)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_term)

    # tell the server whether we want to do ranked matches or not
    if ranked:
        ranked_mode(server, keep_version)
    else:
        unranked_mode(server)

    listen_for_invitations(server, max_concurrent_games)


if __name__ == '__main__':
    main()
