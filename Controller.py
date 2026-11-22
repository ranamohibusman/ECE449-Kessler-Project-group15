from kesslergame import KesslerController


class MyController(KesslerController):

    def __init__(self):
        pass

    @property
    def name(self):
        return "My Fuzzy Controller"

    def actions(self, ship_state, game_state):
        thrust = 0.0
        turn_rate = 0.0
        fire = False
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine