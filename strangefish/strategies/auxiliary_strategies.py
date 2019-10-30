import random


def _choose_randomly(_, __, choices, *args, **kwargs):
    return random.choice(choices)


def _do_nothing(*args, **kwargs):
    pass


def random_strategy():
    choose_sense = _choose_randomly
    choose_move = _choose_randomly
    while_we_wait = _do_nothing
    end_game = _do_nothing
    return choose_sense, choose_move, while_we_wait, end_game


def idle_strategy():
    choose_sense = _do_nothing
    choose_move = _do_nothing
    while_we_wait = _do_nothing
    end_game = _do_nothing
    return choose_sense, choose_move, while_we_wait, end_game


def contingency_strategy():
    choose_sense = _do_nothing
    choose_move = _choose_randomly
    return choose_sense, choose_move
