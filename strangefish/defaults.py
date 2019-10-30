import random


def _choose_randomly(_, __, choices, *args, **kwargs):
    return random.choice(choices)


def _do_nothing(*args, **kwargs):
    pass


choose_move = _choose_randomly
choose_sense = _choose_randomly
while_we_wait = _do_nothing
end_game = _do_nothing
