# Asteroid Annihilator: Group 15 (Rajit Shah, Mohib Usman, Haydn Gareth Wowk)
#
# Four fuzzy systems (skfuzzy, Mamdani) decide everything the ship does:
#   aim  -> turn_rate,  fire -> fire,  move -> thrust,  mine -> drop_mine
# The membership function breakpoints come from a chromosome that the genetic algorithm at the bottom of this file tunes. 

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import time

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy.control.controlsystem import RuleOrderGenerator

from kesslergame import KesslerController, KesslerGame, GraphicsType, Scenario


# Best chromosome found by the GA (each gene is between 0 and 1)
DEFAULT_CHROMOSOME = [0.053, 0.000, 0.449, 0.743, 0.864, 0.490, 0.781, 0.542, 0.482, 0.143, 0.298, 0.415]

# What each gene controls: (name, value at gene=0, value at gene=1)
GENES = [
    ("threat_margin",    0.0,  30.0),  # extra clearance added to collision radii (m)
    ("danger_imminent",  0.3,   1.5),  # danger_time below this is "imminent" (s)
    ("danger_far",       1.2,   2.6),  # danger_time above this is "far" (s)
    ("aim_saturation",   3.0,  12.0),  # aim error that gets a full-speed turn (deg)
    ("shot_good",        0.3,   1.0),  # shot_quality that counts as a sure hit
    ("respawn_hold",     0.2,   2.8),  # hold fire while more respawn time than this is left (s)
    ("mine_many",        2.0,  12.0),  # asteroids in the blast that count as "many"
    ("opp_close",       60.0, 260.0),  # opponent distance that counts as "close" (m)
    ("threat_priority",  0.0,   2.0),  # targeting bonus for asteroids headed at us
    ("bullet_weight",    0.3,   2.5),  # targeting weight of bullet travel time
    ("respawn_flee",     0.4,   2.5),  # danger_time we pretend to have while relocating after a respawn
    ("pref_dead",        0.1,   0.7),  # escape_pref below this means forward and reverse are about equal
]

BULLET_SPEED = 800.0
SHIP_RADIUS = 20.0
MINE_BLAST = 150.0
MINE_FUSE = 3.0
MAX_TURN = 180.0
MAX_THRUST_OUT = 400.0   # strongest thrust the move rules can actually output
LOOKAHEAD_STEP = 0.05
LOOKAHEAD_STEPS = 50     # 2.5 s
NOTHING_IN_SIGHT = 5.0
MINE_ESCAPE_LEAD = 2.0   # warn about a mine this many seconds before it blows


class FastControlSystem(ctrl.ControlSystem):
    # Stock skfuzzy works out the rule order again on every compute()
    # Caching it gives the same answers about 15x faster.
    @property
    def rules(self):
        if "_rule_order" not in self.__dict__:
            self.__dict__["_rule_order"] = RuleOrderGenerator(self)
        return self.__dict__["_rule_order"]


class FuzzySystems:
    def __init__(self, chromosome=None):
        if chromosome is None:
            chromosome = DEFAULT_CHROMOSOME
        if len(chromosome) != len(GENES):
            raise ValueError("chromosome needs %d genes" % len(GENES))

        self.params = {}
        for gene, (name, low, high) in zip(chromosome, GENES):
            gene = min(1.0, max(0.0, float(gene)))
            self.params[name] = low + gene * (high - low)

        self.aim = self.build_aim()
        self.fire = self.build_fire()
        self.move = self.build_move()
        self.mine = self.build_mine()

    def run(self, system, inputs, output):
        for name, value in inputs.items():
            system.input[name] = value
        system.compute()
        return system.output[output]

    def build_aim(self):
        saturation = self.params["aim_saturation"]
        aim = ctrl.Antecedent(np.linspace(-30.0, 30.0, 121), "aim_error")
        turn = ctrl.Consequent(np.linspace(-210.0, 210.0, 85), "turn_rate")

        aim["NL"] = fuzz.trapmf(aim.universe, [-30, -30, -saturation, -2 * saturation / 3])
        aim["NM"] = fuzz.trimf(aim.universe, [-saturation, -2 * saturation / 3, -saturation / 3])
        aim["NS"] = fuzz.trimf(aim.universe, [-2 * saturation / 3, -saturation / 3, 0])
        aim["Z"] = fuzz.trimf(aim.universe, [-saturation / 3, 0, saturation / 3])
        aim["PS"] = fuzz.trimf(aim.universe, [0, saturation / 3, 2 * saturation / 3])
        aim["PM"] = fuzz.trimf(aim.universe, [saturation / 3, 2 * saturation / 3, saturation])
        aim["PL"] = fuzz.trapmf(aim.universe, [2 * saturation / 3, saturation, 30, 30])

        turn["NL"] = fuzz.trimf(turn.universe, [-210, -210, -150])
        turn["NM"] = fuzz.trimf(turn.universe, [-180, -120, -60])
        turn["NS"] = fuzz.trimf(turn.universe, [-120, -60, 0])
        turn["Z"] = fuzz.trimf(turn.universe, [-60, 0, 60])
        turn["PS"] = fuzz.trimf(turn.universe, [0, 60, 120])
        turn["PM"] = fuzz.trimf(turn.universe, [60, 120, 180])
        turn["PL"] = fuzz.trimf(turn.universe, [150, 210, 210])

        # turn the same way, and about as hard, as we are off target
        rules = [ctrl.Rule(aim[term], turn[term]) for term in ("NL", "NM", "NS", "Z", "PS", "PM", "PL")]
        return ctrl.ControlSystemSimulation(FastControlSystem(rules), cache=False)

    def build_fire(self):
        good = self.params["shot_good"]
        hold = self.params["respawn_hold"]
        quality = ctrl.Antecedent(np.linspace(0.0, 3.0, 61), "shot_quality")
        respawn = ctrl.Antecedent(np.linspace(0.0, 3.0, 31), "respawn_left")
        fire = ctrl.Consequent(np.linspace(-1.0, 1.0, 21), "fire")

        # shot_quality = how far the bullet would miss the asteroid's centre, in asteroid radii
        quality["good"] = fuzz.trapmf(quality.universe, [0, 0, good, good + 0.25])
        quality["fair"] = fuzz.trimf(quality.universe, [good, good + 0.25, 1.4])
        quality["poor"] = fuzz.trapmf(quality.universe, [good + 0.25, 1.4, 3, 3])

        respawn["none"] = fuzz.trapmf(respawn.universe, [0, 0, max(0.05, hold - 0.2), hold])
        respawn["long"] = fuzz.trapmf(respawn.universe, [max(0.05, hold - 0.2), hold, 3, 3])

        fire["no"] = fuzz.trimf(fire.universe, [-1, -1, 0])
        fire["yes"] = fuzz.trimf(fire.universe, [0, 1, 1])

        rules = [
            ctrl.Rule(quality["good"] & respawn["none"], fire["yes"]),
            ctrl.Rule(quality["fair"] & respawn["none"], fire["no"]),
            ctrl.Rule(quality["poor"], fire["no"]),
            ctrl.Rule(respawn["long"], fire["no"]),  # firing would end our invincibility
        ]
        return ctrl.ControlSystemSimulation(FastControlSystem(rules), cache=False)

    def build_move(self):
        imminent = self.params["danger_imminent"]
        far = max(self.params["danger_far"], imminent + 0.3)
        mid = 0.5 * (imminent + far)
        dead_band = self.params["pref_dead"]

        danger = ctrl.Antecedent(np.linspace(0.0, 5.0, 51), "danger_time")
        pref = ctrl.Antecedent(np.linspace(-1.0, 1.0, 41), "escape_pref")
        speed = ctrl.Antecedent(np.linspace(-240.0, 240.0, 49), "speed")
        thrust = ctrl.Consequent(np.linspace(-480.0, 480.0, 49), "thrust")

        danger["imminent"] = fuzz.trapmf(danger.universe, [0, 0, imminent, mid])
        danger["near"] = fuzz.trimf(danger.universe, [imminent, mid, far])
        danger["far"] = fuzz.trapmf(danger.universe, [mid, far, 5, 5])

        # escape_pref > 0: thrusting forward buys more time, < 0: reversing does
        pref["back"] = fuzz.trapmf(pref.universe, [-1, -1, -dead_band - 0.2, -dead_band + 0.05])
        pref["side"] = fuzz.trimf(pref.universe, [-dead_band - 0.05, 0, dead_band + 0.05])
        pref["front"] = fuzz.trapmf(pref.universe, [dead_band - 0.05, dead_band + 0.2, 1, 1])

        speed["rev"] = fuzz.trapmf(speed.universe, [-240, -240, -60, -5])
        speed["still"] = fuzz.trimf(speed.universe, [-40, 0, 40])
        speed["fwd"] = fuzz.trapmf(speed.universe, [5, 60, 240, 240])

        thrust["rev_hard"] = fuzz.trimf(thrust.universe, [-480, -480, -240])
        thrust["rev_soft"] = fuzz.trimf(thrust.universe, [-400, -200, 0])
        thrust["zero"] = fuzz.trimf(thrust.universe, [-120, 0, 120])
        thrust["fwd_soft"] = fuzz.trimf(thrust.universe, [0, 200, 400])
        thrust["fwd_hard"] = fuzz.trimf(thrust.universe, [240, 480, 480])

        rules = [
            # about to get hit: get out of the way
            ctrl.Rule(danger["imminent"] & pref["front"], thrust["fwd_hard"]),
            ctrl.Rule(danger["imminent"] & pref["back"], thrust["rev_hard"]),
            ctrl.Rule(danger["imminent"] & pref["side"] & speed["fwd"], thrust["fwd_soft"]),
            ctrl.Rule(danger["imminent"] & pref["side"] & speed["rev"], thrust["rev_soft"]),
            ctrl.Rule(danger["imminent"] & pref["side"] & speed["still"], thrust["fwd_soft"]),
            # something coming: ease away
            ctrl.Rule(danger["near"] & pref["front"], thrust["fwd_soft"]),
            ctrl.Rule(danger["near"] & pref["back"], thrust["rev_soft"]),
            ctrl.Rule(danger["near"] & pref["side"], thrust["zero"]),
            # all clear: brake and sit still so we shoot accurately
            ctrl.Rule(danger["far"] & speed["fwd"], thrust["rev_soft"]),
            ctrl.Rule(danger["far"] & speed["rev"], thrust["fwd_soft"]),
            ctrl.Rule(danger["far"] & speed["still"], thrust["zero"]),
        ]
        return ctrl.ControlSystemSimulation(FastControlSystem(rules), cache=False)

    def build_mine(self):
        many = self.params["mine_many"]
        close = self.params["opp_close"]
        value = ctrl.Antecedent(np.linspace(0.0, 20.0, 41), "mine_value")
        opp_near = ctrl.Antecedent(np.linspace(0.0, 1000.0, 51), "opp_distance")
        drop = ctrl.Consequent(np.linspace(-1.0, 1.0, 21), "drop_mine")

        # mine_value = how many asteroids will be inside the blast when it goes off
        value["few"] = fuzz.trapmf(value.universe, [0, 0, many * 0.35, many * 0.7])
        value["some"] = fuzz.trimf(value.universe, [many * 0.35, many * 0.7, many])
        value["many"] = fuzz.trapmf(value.universe, [many * 0.7, many, 20, 20])

        opp_near["close"] = fuzz.trapmf(opp_near.universe, [0, 0, close * 0.6, close])
        opp_near["far"] = fuzz.trapmf(opp_near.universe, [close * 0.6, close, 1000, 1000])

        drop["no"] = fuzz.trimf(drop.universe, [-1, -1, 0])
        drop["yes"] = fuzz.trimf(drop.universe, [0, 1, 1])

        rules = [
            ctrl.Rule(value["many"], drop["yes"]),
            ctrl.Rule(value["some"] & opp_near["close"], drop["yes"]),
            ctrl.Rule(value["some"] & opp_near["far"], drop["no"]),
            ctrl.Rule(value["few"], drop["no"]),
        ]
        return ctrl.ControlSystemSimulation(FastControlSystem(rules), cache=False)


class AsteroidAnnihilator(KesslerController):
    def __init__(self, chromosome=None):
        self.eval_frames = 0
        self.fuzzy = FuzzySystems(chromosome)
        self.params = self.fuzzy.params
        self.lookahead_times = LOOKAHEAD_STEP * np.arange(1, LOOKAHEAD_STEPS + 1)

    @property
    def name(self) -> str:
        return "Asteroid Annihilator"

    def time_to_impact(self, thrust, ship_x, ship_y, speed, heading, turn_rate, turn_time,
                       ast_x, ast_y, ast_vel_x, ast_vel_y, ast_radius, opponents, mines, map_width, map_height):
        """How many seconds until we hit something if we hold this thrust.
        Simulates the ship (drag, speed cap, still turning toward its target)
        and checks asteroids, the other ship and mine blasts."""
        times = self.lookahead_times
        time_step = LOOKAHEAD_STEP

        # where the ship will be at each look-ahead time
        path_x = np.empty(len(times))
        path_y = np.empty(len(times))
        pos_x, pos_y, speed_now = ship_x, ship_y, speed
        for step_index in range(len(times)):
            if abs(speed_now) > 1e-9:
                if thrust == 0.0 and abs(speed_now) <= 80.0 * time_step:
                    speed_next = 0.0
                else:
                    speed_next = speed_now + (thrust - 80.0 * math.copysign(1.0, speed_now)) * time_step
            else:
                speed_next = speed_now + (thrust - 80.0 * math.copysign(1.0, thrust)) * time_step if thrust != 0.0 else 0.0
            speed_next = max(-240.0, min(240.0, speed_next))

            # velocity always points where the ship is facing
            angle = math.radians(heading + turn_rate * min(times[step_index] - 0.5 * time_step, turn_time))
            step_dist = 0.5 * (speed_now + speed_next) * time_step
            pos_x += step_dist * math.cos(angle)
            pos_y += step_dist * math.sin(angle)
            path_x[step_index] = pos_x
            path_y[step_index] = pos_y
            speed_now = speed_next

        margin = self.params["threat_margin"]
        first_hit = NOTHING_IN_SIGHT

        if len(ast_x):
            gap_x = ast_x[None, :] + ast_vel_x[None, :] * times[:, None] - path_x[:, None]
            gap_y = ast_y[None, :] + ast_vel_y[None, :] * times[:, None] - path_y[:, None]
            gap_x = (gap_x + map_width / 2) % map_width - map_width / 2
            gap_y = (gap_y + map_height / 2) % map_height - map_height / 2
            touching = (gap_x ** 2 + gap_y ** 2) < ((ast_radius + SHIP_RADIUS + margin) ** 2)[None, :]
            hit_steps = np.nonzero(touching.any(axis=1))[0]
            if len(hit_steps):
                first_hit = float(times[hit_steps[0]])

            gap_now_x = (ast_x - ship_x + map_width / 2) % map_width - map_width / 2
            gap_now_y = (ast_y - ship_y + map_height / 2) % map_height - map_height / 2
            if np.any(gap_now_x ** 2 + gap_now_y ** 2 < (ast_radius + SHIP_RADIUS) ** 2):
                first_hit = 0.0

        for other in opponents:
            if other["is_respawning"]:
                continue
            other_x, other_y = other["position"]
            other_vel_x, other_vel_y = other["velocity"]
            gap_x = (other_x + other_vel_x * times - path_x + map_width / 2) % map_width - map_width / 2
            gap_y = (other_y + other_vel_y * times - path_y + map_height / 2) % map_height - map_height / 2
            hit_steps = np.nonzero(gap_x ** 2 + gap_y ** 2 < (2 * SHIP_RADIUS + margin + 10.0) ** 2)[0]
            if len(hit_steps):
                first_hit = min(first_hit, float(times[hit_steps[0]]))

        for mine in mines:
            fuse = float(mine["remaining_time"])
            step_index = min(len(times) - 1, max(0, int(round(fuse / time_step)) - 1))
            if fuse > times[-1]:
                step_index = len(times) - 1
            mine_x, mine_y = mine["position"]
            # the blast doesn't wrap, so use plain distance to where we'll be on the map
            gap_x = mine_x - path_x[step_index] % map_width
            gap_y = mine_y - path_y[step_index] % map_height
            if gap_x ** 2 + gap_y ** 2 < (MINE_BLAST + SHIP_RADIUS + 15.0) ** 2:
                # warn early so there is still time to run
                first_hit = min(first_hit, max(0.0, fuse - MINE_ESCAPE_LEAD))

        return first_hit

    def actions(self, ship_state, game_state):
        # make sure numpy never prints a warning during a game
        with np.errstate(all="ignore"):
            return self.decide(ship_state, game_state)

    def decide(self, ship_state, game_state):
        self.eval_frames += 1
        params = self.params
        fuzzy = self.fuzzy

        map_width, map_height = game_state["map_size"]
        frame_time = game_state["delta_time"]
        ship_x, ship_y = ship_state["position"]
        ship_vel_x, ship_vel_y = ship_state["velocity"]
        heading = ship_state["heading"]
        speed = ship_state["speed"]
        respawning = bool(ship_state["is_respawning"])
        respawn_left = float(ship_state["respawn_time_left"]) if respawning else 0.0

        asteroids = game_state["asteroids"]
        num_asteroids = len(asteroids)
        if num_asteroids:
            ast_data = np.array([(ast["position"][0], ast["position"][1], ast["velocity"][0], ast["velocity"][1], ast["radius"])
                                 for ast in asteroids], dtype=float)
            ast_x, ast_y, ast_vel_x, ast_vel_y, ast_radius = ast_data.T
        else:
            ast_x = ast_y = ast_vel_x = ast_vel_y = ast_radius = np.zeros(0)

        opponents = [other for other in game_state["ships"] if other["id"] != ship_state["id"]]
        mines = game_state["mines"]
        bullets = game_state["bullets"]

        # How soon each asteroid hits us if we just coast. Offsets wrap around the screen.
        time_to_collision = np.full(num_asteroids, np.inf)
        if num_asteroids:
            ast_offset_x = (ast_x - ship_x + map_width / 2) % map_width - map_width / 2
            ast_offset_y = (ast_y - ship_y + map_height / 2) % map_height - map_height / 2
            relative_vel_x = ast_vel_x - ship_vel_x
            relative_vel_y = ast_vel_y - ship_vel_y
            reach = ast_radius + SHIP_RADIUS + params["threat_margin"]
            quad_a = relative_vel_x * relative_vel_x + relative_vel_y * relative_vel_y
            quad_b = 2.0 * (ast_offset_x * relative_vel_x + ast_offset_y * relative_vel_y)
            quad_c = ast_offset_x * ast_offset_x + ast_offset_y * ast_offset_y - reach * reach
            discriminant = quad_b * quad_b - 4.0 * quad_a * quad_c
            time_until_hit = (-quad_b - np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * quad_a)
            will_hit = (discriminant >= 0) & (quad_a > 1e-9) & (time_until_hit >= 0)
            time_to_collision = np.where(quad_c <= 0, 0.0, np.where(will_hit, time_until_hit, np.inf))

        opp_dist = 1000.0
        for other in opponents:
            other_x, other_y = other["position"]
            gap_x = (other_x - ship_x + map_width / 2) % map_width - map_width / 2
            gap_y = (other_y - ship_y + map_height / 2) % map_height - map_height / 2
            opp_dist = min(opp_dist, math.hypot(gap_x, gap_y))

        # Asteroids that a bullet already in flight is going to destroy
        doomed = np.zeros(num_asteroids, dtype=bool)
        if num_asteroids and len(bullets):
            bullet_data = np.array([(bullet["position"][0], bullet["position"][1], bullet["velocity"][0], bullet["velocity"][1])
                                    for bullet in bullets], dtype=float)
            bullet_x, bullet_y, bullet_vel_x, bullet_vel_y = bullet_data[:, 0:1], bullet_data[:, 1:2], bullet_data[:, 2:3], bullet_data[:, 3:4]
            gap_x = ast_x[None, :] - bullet_x
            gap_y = ast_y[None, :] - bullet_y
            relative_vel_x = ast_vel_x[None, :] - bullet_vel_x
            relative_vel_y = ast_vel_y[None, :] - bullet_vel_y
            rel_speed_sq = relative_vel_x * relative_vel_x + relative_vel_y * relative_vel_y
            time_of_closest_approach = np.clip(-(gap_x * relative_vel_x + gap_y * relative_vel_y) / np.maximum(rel_speed_sq, 1e-9), 0.0, 2.0)
            miss_dist_x = gap_x + relative_vel_x * time_of_closest_approach
            miss_dist_y = gap_y + relative_vel_y * time_of_closest_approach
            hit = (miss_dist_x * miss_dist_x + miss_dist_y * miss_dist_y) < (ast_radius[None, :] + 1.0) ** 2
            # bullets disappear when they leave the map
            end_x = bullet_x + bullet_vel_x * time_of_closest_approach
            end_y = bullet_y + bullet_vel_y * time_of_closest_approach
            hit &= (end_x >= 0) & (end_x <= map_width) & (end_y >= 0) & (end_y <= map_height)
            # each bullet only kills the first asteroid it reaches
            time_until_hit = np.where(hit, time_of_closest_approach, np.inf)
            first = np.argmin(time_until_hit, axis=1)
            got_one = np.isfinite(time_until_hit[np.arange(len(first)), first])
            doomed[first[got_one]] = True

        # Pick a target: the asteroid we can turn to and hit soonest.
        # Every asteroid is tried at all 9 screen-wrapped positions.
        aim_error = 0.0
        if num_asteroids:
            shifts = np.array([(shift_x, shift_y) for shift_x in (-map_width, 0.0, map_width) for shift_y in (-map_height, 0.0, map_height)])
            target_x = (ast_x[None, :] + shifts[:, 0:1]).ravel()
            target_y = (ast_y[None, :] + shifts[:, 1:2]).ravel()
            target_vel_x = np.tile(ast_vel_x, 9)
            target_vel_y = np.tile(ast_vel_y, 9)
            target_radius = np.tile(ast_radius, 9)
            target_doomed = np.tile(doomed, 9)
            target_time_to_collision = np.tile(time_to_collision, 9)

            turn_time = np.zeros_like(target_x)
            for _ in range(2):  # second pass accounts for the time spent turning
                fire_delay = turn_time + frame_time
                fire_x = ship_x + ship_vel_x * fire_delay
                fire_y = ship_y + ship_vel_y * fire_delay
                target_offset_x = target_x + target_vel_x * fire_delay - fire_x
                target_offset_y = target_y + target_vel_y * fire_delay - fire_y
                quad_a = target_vel_x * target_vel_x + target_vel_y * target_vel_y - BULLET_SPEED ** 2
                quad_b = 2.0 * (target_offset_x * target_vel_x + target_offset_y * target_vel_y) - 2.0 * BULLET_SPEED * SHIP_RADIUS
                quad_c = target_offset_x * target_offset_x + target_offset_y * target_offset_y - SHIP_RADIUS ** 2
                discriminant = np.maximum(quad_b * quad_b - 4.0 * quad_a * quad_c, 0.0)
                # a == 0 only if an asteroid moves at exactly bullet speed; then it's linear
                linear = np.abs(quad_a) < 1e-6
                safe_quad_a = np.where(linear, -1.0, quad_a)
                root1 = (-quad_b + np.sqrt(discriminant)) / (2.0 * safe_quad_a)
                root2 = (-quad_b - np.sqrt(discriminant)) / (2.0 * safe_quad_a)
                root_linear = np.where(np.abs(quad_b) > 1e-9, -quad_c / np.where(np.abs(quad_b) > 1e-9, quad_b, 1.0), np.inf)
                bullet_time = np.where(linear, root_linear, np.maximum(root1, root2))
                bullet_time = np.where(np.isfinite(bullet_time), np.maximum(bullet_time, 0.0), 1e9)
                hit_x = target_offset_x + target_vel_x * bullet_time
                hit_y = target_offset_y + target_vel_y * bullet_time
                angle = np.degrees(np.arctan2(hit_y, hit_x))
                angle_diff = (angle - heading + 180.0) % 360.0 - 180.0
                turn_time = np.abs(angle_diff) / MAX_TURN

            # the intercept has to be on the map (bullets don't wrap)
            world_x = hit_x + fire_x
            world_y = hit_y + fire_y
            on_map = ((world_x > target_radius) & (world_x < map_width - target_radius)
                      & (world_y > target_radius) & (world_y < map_height - target_radius))
            valid = on_map & ~target_doomed & (bullet_time < 1.4)
            if valid.any():
                cost = turn_time + params["bullet_weight"] * bullet_time - params["threat_priority"] * np.exp(-target_time_to_collision)
                cost = np.where(valid, cost, np.inf)
                aim_error = float(angle_diff[int(np.argmin(cost))])

        turn_rate = fuzzy.run(fuzzy.aim, {"aim_error": float(min(max(aim_error, -30.0), 30.0))}, "turn_rate")
        turn_rate = float(min(max(turn_rate, -MAX_TURN), MAX_TURN))

        # Movement: how long until we hit something if we go forward, coast, or reverse?
        turn_time = min(abs(aim_error) / MAX_TURN, LOOKAHEAD_STEP * LOOKAHEAD_STEPS)
        time_safe_forward, time_safe_coast, time_safe_reverse = (
            self.time_to_impact(plan_thrust, ship_x, ship_y, speed, heading, turn_rate, turn_time,
                                ast_x, ast_y, ast_vel_x, ast_vel_y, ast_radius, opponents, mines, map_width, map_height)
            for plan_thrust in (MAX_THRUST_OUT, 0.0, -MAX_THRUST_OUT))
        danger_time = time_safe_coast
        best_escape_time = max(time_safe_forward, time_safe_reverse)
        escape_pref = (time_safe_forward - time_safe_reverse) / best_escape_time if best_escape_time > 1e-6 else 0.0

        mine_near = any(math.hypot(mine["position"][0] - ship_x, mine["position"][1] - ship_y) < MINE_BLAST + SHIP_RADIUS + 30.0
                        for mine in mines)

        # Just respawned and invincible: move away from nearby asteroids
        if respawning and respawn_left > 0.5 and num_asteroids:
            dist_squared = ast_offset_x * ast_offset_x + ast_offset_y * ast_offset_y + 1.0
            near = dist_squared < 300.0 ** 2
            if near.any():
                away_x = -float(np.sum(ast_offset_x[near] / dist_squared[near]))
                away_y = -float(np.sum(ast_offset_y[near] / dist_squared[near]))
                length = math.hypot(away_x, away_y)
                if length > 1e-9:
                    angle_rad = math.radians(heading)
                    escape_pref = (away_x * math.cos(angle_rad) + away_y * math.sin(angle_rad)) / length
                    danger_time = min(danger_time, params["respawn_flee"])

        thrust = fuzzy.run(fuzzy.move, {"danger_time": float(min(max(danger_time, 0.0), 5.0)),
                                        "escape_pref": float(min(max(escape_pref, -1.0), 1.0)),
                                        "speed": float(min(max(speed, -240.0), 240.0))}, "thrust")

        # Shot quality: simulate the bullet we would fire this frame
        shot_quality = 3.0
        if num_asteroids:
            angle_rad = math.radians(heading + turn_rate * frame_time)
            dir_x, dir_y = math.cos(angle_rad), math.sin(angle_rad)
            start_x = ship_x + ship_vel_x * frame_time + SHIP_RADIUS * dir_x
            start_y = ship_y + ship_vel_y * frame_time + SHIP_RADIUS * dir_y
            bullet_vel_x, bullet_vel_y = BULLET_SPEED * dir_x, BULLET_SPEED * dir_y
            gap_x = ast_x + ast_vel_x * frame_time - start_x
            gap_y = ast_y + ast_vel_y * frame_time - start_y
            relative_vel_x = ast_vel_x - bullet_vel_x
            relative_vel_y = ast_vel_y - bullet_vel_y
            rel_speed_sq = relative_vel_x * relative_vel_x + relative_vel_y * relative_vel_y
            time_of_closest_approach = np.clip(-(gap_x * relative_vel_x + gap_y * relative_vel_y) / np.maximum(rel_speed_sq, 1e-9), 0.0, 1.6)
            miss_dist_x = gap_x + relative_vel_x * time_of_closest_approach
            miss_dist_y = gap_y + relative_vel_y * time_of_closest_approach
            miss_ratio = np.sqrt(miss_dist_x * miss_dist_x + miss_dist_y * miss_dist_y) / ast_radius
            end_x = start_x + bullet_vel_x * time_of_closest_approach
            end_y = start_y + bullet_vel_y * time_of_closest_approach
            usable = (end_x >= 0) & (end_x <= map_width) & (end_y >= 0) & (end_y <= map_height) & ~doomed
            hits = usable & (miss_ratio < 1.0)
            if hits.any():
                # whichever asteroid the bullet reaches first is the one that counts
                shot_quality = float(miss_ratio[int(np.argmin(np.where(hits, time_of_closest_approach, np.inf)))])
            elif usable.any():
                shot_quality = float(min(3.0, miss_ratio[usable].min()))

        fire_out = fuzzy.run(fuzzy.fire, {"shot_quality": float(min(shot_quality, 3.0)),
                                          "respawn_left": float(min(respawn_left, 3.0))}, "fire")
        fire = bool(fire_out > 0.0)

        # Mines: only if no other mine is about to go off near us and we have a way out
        drop_mine = False
        if (ship_state["mines_remaining"] != 0 and ship_state["can_deploy_mine"] and not mine_near and num_asteroids
                and max(time_safe_forward, time_safe_reverse) >= MINE_ESCAPE_LEAD):
            # asteroids wrap but the blast doesn't, so use plain distance here
            future_x = (ast_x + ast_vel_x * MINE_FUSE) % map_width
            future_y = (ast_y + ast_vel_y * MINE_FUSE) % map_height
            in_blast = ((future_x - ship_x) ** 2 + (future_y - ship_y) ** 2) < (MINE_BLAST + ast_radius) ** 2
            mine_value = float(np.count_nonzero(in_blast & ~doomed))
            mine_out = fuzzy.run(fuzzy.mine, {"mine_value": min(mine_value, 20.0),
                                              "opp_distance": float(min(opp_dist, 1000.0))}, "drop_mine")
            drop_mine = bool(mine_out > 0.0)
            if respawning and respawn_left > 0.5:
                drop_mine = False

        return float(thrust), turn_rate, fire, drop_mine


# ---------------------------------------------------------------------------
# Genetic algorithm (training only)
#
# fitness = average over games of (our hits - their hits) + 20 * our accuracy
# Each match is 5 games that get harder (more and faster asteroids). One of the
# five is against a dummy that rams us, so we also learn to avoid the other ship.
# ---------------------------------------------------------------------------

MAP_SIZE = (1000, 800)
START_POS = [(400, 400), (600, 400)]


class Rammer(KesslerController):
    """Training dummy that chases the other ship."""

    def __init__(self):
        self.eval_frames = 0

    def actions(self, ship_state, game_state):
        self.eval_frames += 1
        sx, sy = ship_state["position"]
        W, H = game_state["map_size"]
        others = [s for s in game_state["ships"] if s["id"] != ship_state["id"]]
        if not others:
            return 0.0, 90.0, True, False
        ox, oy = others[0]["position"]
        dx = (ox - sx + W / 2) % W - W / 2
        dy = (oy - sy + H / 2) % H - H / 2
        err = (math.degrees(math.atan2(dy, dx)) - ship_state["heading"] + 180) % 360 - 180
        thrust = 300.0 if abs(err) < 30 else 0.0
        return thrust, max(-180.0, min(180.0, err * 10)), True, False

    @property
    def name(self):
        return "Rammer Dummy"


def make_scenario(game_idx, rng, time_limit=60, swap=False, n_games=5):
    level = game_idx / max(1, n_games - 1)
    count = int(round(8 + 17 * level))   # 8 to 25 asteroids
    max_speed = 60 + 120 * level         # 60 to 180 m/s
    states = []
    while len(states) < count:
        x = rng.uniform(0, MAP_SIZE[0])
        y = rng.uniform(0, MAP_SIZE[1])
        if any((x - px) ** 2 + (y - py) ** 2 < 130 ** 2 for px, py in START_POS):
            continue  # keep the spawn points clear
        states.append({"position": (x, y), "angle": rng.uniform(0, 360),
                       "speed": rng.uniform(0.3, 1.0) * max_speed,
                       "size": rng.choice([2, 3, 3, 4, 4])})
    pos = START_POS[::-1] if swap else START_POS
    return Scenario(name="GA game %d" % (game_idx + 1), asteroid_states=states,
                    ship_states=[
                        {"position": pos[0], "angle": 90, "lives": 3, "team": 1, "mines_remaining": 3},
                        {"position": pos[1], "angle": 90, "lives": 3, "team": 2, "mines_remaining": 3},
                    ],
                    map_size=MAP_SIZE, time_limit=time_limit,
                    ammo_limit_multiplier=0, stop_if_no_ammo=False)


def play_match(chromosome, seed, games=5, rammer_game=2):
    from test_controller import TestController

    game = KesslerGame(settings={"graphics_type": GraphicsType.NoGraphics,
                                 "realtime_multiplier": 0, "prints_on": False})
    rng = random.Random(seed)
    margin, accs, hits, deaths = 0.0, [], 0, 0
    for g in range(games):
        opponent = Rammer() if g == rammer_game else TestController()
        score, _ = game.run(scenario=make_scenario(g, rng, swap=bool(seed % 2)),
                            controllers=[AsteroidAnnihilator(chromosome), opponent])
        us, them = score.teams[0], score.teams[1]
        # the rammer isn't trying to score, so only our hits count in that game
        margin += us.asteroids_hit - (0 if g == rammer_game else them.asteroids_hit)
        accs.append(us.accuracy)
        hits += us.asteroids_hit
        deaths += us.deaths
    return margin, accs, hits, deaths


def fitness(args):
    chromosome, seeds = args
    margin, accs, hits, deaths = 0.0, [], 0, 0
    for seed in seeds:
        m, a, h, d = play_match(chromosome, seed)
        margin += m
        accs += a
        hits += h
        deaths += d
    return margin / len(accs) + 20.0 * (sum(accs) / len(accs)), hits, deaths


def tournament_select(pop, fits, k=3):
    return pop[max(random.sample(range(len(pop)), k), key=lambda i: fits[i])]


def blend_crossover(a, b, alpha=0.3):
    # BLX-alpha: pick each child gene from a range slightly wider than the parents'
    child = []
    for x, y in zip(a, b):
        low, high = min(x, y), max(x, y)
        span = high - low
        child.append(min(1.0, max(0.0, random.uniform(low - alpha * span, high + alpha * span))))
    return child


def mutate(c, rate=0.2, sigma=0.12):
    return [min(1.0, max(0.0, g + random.gauss(0, sigma))) if random.random() < rate else g for g in c]


def bake(chromosome, target):
    with open(target) as f:
        src = f.read()
    line = "DEFAULT_CHROMOSOME = [" + ", ".join(repr(float(g)) for g in chromosome) + "]"
    src, count = re.subn(r"^DEFAULT_CHROMOSOME = \[.*\]$", line, src, count=1, flags=re.M)
    if count == 0:
        raise SystemExit("--bake: no DEFAULT_CHROMOSOME line found in " + target)
    with open(target, "w") as f:
        f.write(src)
    return line


def main():
    parser = argparse.ArgumentParser(description="Tune the fuzzy agent with a genetic algorithm.")
    parser.add_argument("--pop", type=int, default=16, help="population size")
    parser.add_argument("--gens", type=int, default=12, help="number of generations")
    parser.add_argument("--matches", type=int, default=1, help="5-game matches per fitness evaluation")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="parallel processes (default: CPU cores - 1)")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for the GA")
    parser.add_argument("--out", default="best_chromosome.json", help="where to save the result")
    parser.add_argument("--bake", action="store_true", help="write the winner into --bake-target")
    parser.add_argument("--bake-target", default=os.path.abspath(__file__),
                        help="file with the DEFAULT_CHROMOSOME line (default: this file)")
    args = parser.parse_args()

    random.seed(args.seed)

    # start from the current best plus some mutated copies, fill the rest randomly
    pop = [list(DEFAULT_CHROMOSOME)]
    pop += [mutate(list(DEFAULT_CHROMOSOME), rate=0.5, sigma=0.15) for _ in range(args.pop // 3)]
    while len(pop) < args.pop:
        pop.append([random.random() for _ in range(len(GENES))])
    pop = pop[:args.pop]

    history = []
    with mp.Pool(args.workers) as pool:
        for gen in range(args.gens):
            start = time.time()
            # new random fields every generation, the same ones for everybody
            seeds = [random.randrange(10 ** 6) for _ in range(args.matches)]
            results = pool.map(fitness, [(c, seeds) for c in pop])
            fits = [r[0] for r in results]
            order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
            best = order[0]
            history.append({"gen": gen, "best": fits[best], "mean": sum(fits) / len(fits),
                            "chromosome": pop[best]})
            print("gen %2d  best %7.2f  mean %7.2f  (hits %d, deaths %d)  %.0fs"
                  % (gen, fits[best], sum(fits) / len(fits), results[best][1], results[best][2],
                     time.time() - start), flush=True)

            children = [pop[i] for i in order[:2]]  # keep the top 2 as they are
            while len(children) < args.pop:
                children.append(mutate(blend_crossover(tournament_select(pop, fits),
                                                       tournament_select(pop, fits))))
            pop = children

        # re-test the best candidates on fields none of them have seen
        print("Validating finalists on fresh fields...", flush=True)
        finalists = pop[:2] + [h["chromosome"] for h in sorted(history, key=lambda h: -h["best"])[:4]]
        finalists.append(list(DEFAULT_CHROMOSOME))
        val_seeds = [random.randrange(10 ** 6) for _ in range(4)]
        val = pool.map(fitness, [(c, val_seeds) for c in finalists])

    k = max(range(len(finalists)), key=lambda i: val[i][0])
    winner = finalists[k]
    print("Winner (validation fitness %.2f):" % val[k][0], ["%.3f" % g for g in winner])
    with open(args.out, "w") as f:
        json.dump({"chromosome": winner, "validation_fitness": val[k][0], "history": history}, f, indent=2)
    print("Saved", args.out)
    if args.bake:
        print("Baked into %s: %s" % (args.bake_target, bake(winner, args.bake_target)))


if __name__ == "__main__":
    main()
