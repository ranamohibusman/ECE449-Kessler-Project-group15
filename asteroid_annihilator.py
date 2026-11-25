# Asteroid Annihilator
#
# Four fuzzy systems (skfuzzy, Mamdani) decide everything the ship does:
#   aim  -> turn_rate,  fire -> fire,  move -> thrust,  mine -> drop_mine
# The membership function breakpoints come from a chromosome that the genetic
# algorithm at the bottom of this file tunes. The GA only runs when you run this
# file directly, e.g. python asteroid_annihilator.py --pop 16 --gens 14 --matches 2 --bake

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
    # Stock skfuzzy works out the rule order again on every compute() (~4 ms).
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
        s = self.params["aim_saturation"]
        aim = ctrl.Antecedent(np.linspace(-30.0, 30.0, 121), "aim_error")
        turn = ctrl.Consequent(np.linspace(-210.0, 210.0, 85), "turn_rate")

        aim["NL"] = fuzz.trapmf(aim.universe, [-30, -30, -s, -2 * s / 3])
        aim["NM"] = fuzz.trimf(aim.universe, [-s, -2 * s / 3, -s / 3])
        aim["NS"] = fuzz.trimf(aim.universe, [-2 * s / 3, -s / 3, 0])
        aim["Z"] = fuzz.trimf(aim.universe, [-s / 3, 0, s / 3])
        aim["PS"] = fuzz.trimf(aim.universe, [0, s / 3, 2 * s / 3])
        aim["PM"] = fuzz.trimf(aim.universe, [s / 3, 2 * s / 3, s])
        aim["PL"] = fuzz.trapmf(aim.universe, [2 * s / 3, s, 30, 30])

        turn["NL"] = fuzz.trimf(turn.universe, [-210, -210, -150])
        turn["NM"] = fuzz.trimf(turn.universe, [-180, -120, -60])
        turn["NS"] = fuzz.trimf(turn.universe, [-120, -60, 0])
        turn["Z"] = fuzz.trimf(turn.universe, [-60, 0, 60])
        turn["PS"] = fuzz.trimf(turn.universe, [0, 60, 120])
        turn["PM"] = fuzz.trimf(turn.universe, [60, 120, 180])
        turn["PL"] = fuzz.trimf(turn.universe, [150, 210, 210])

        # turn the same way, and about as hard, as we are off target
        rules = [ctrl.Rule(aim[t], turn[t]) for t in ("NL", "NM", "NS", "Z", "PS", "PM", "PL")]
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
        dead = self.params["pref_dead"]

        danger = ctrl.Antecedent(np.linspace(0.0, 5.0, 51), "danger_time")
        pref = ctrl.Antecedent(np.linspace(-1.0, 1.0, 41), "escape_pref")
        speed = ctrl.Antecedent(np.linspace(-240.0, 240.0, 49), "speed")
        thrust = ctrl.Consequent(np.linspace(-480.0, 480.0, 49), "thrust")

        danger["imminent"] = fuzz.trapmf(danger.universe, [0, 0, imminent, mid])
        danger["near"] = fuzz.trimf(danger.universe, [imminent, mid, far])
        danger["far"] = fuzz.trapmf(danger.universe, [mid, far, 5, 5])

        # escape_pref > 0: thrusting forward buys more time, < 0: reversing does
        pref["back"] = fuzz.trapmf(pref.universe, [-1, -1, -dead - 0.2, -dead + 0.05])
        pref["side"] = fuzz.trimf(pref.universe, [-dead - 0.05, 0, dead + 0.05])
        pref["front"] = fuzz.trapmf(pref.universe, [dead - 0.05, dead + 0.2, 1, 1])

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
        opp = ctrl.Antecedent(np.linspace(0.0, 1000.0, 51), "opp_distance")
        drop = ctrl.Consequent(np.linspace(-1.0, 1.0, 21), "drop_mine")

        # mine_value = how many asteroids will be inside the blast when it goes off
        value["few"] = fuzz.trapmf(value.universe, [0, 0, many * 0.35, many * 0.7])
        value["some"] = fuzz.trimf(value.universe, [many * 0.35, many * 0.7, many])
        value["many"] = fuzz.trapmf(value.universe, [many * 0.7, many, 20, 20])

        opp["close"] = fuzz.trapmf(opp.universe, [0, 0, close * 0.6, close])
        opp["far"] = fuzz.trapmf(opp.universe, [close * 0.6, close, 1000, 1000])

        drop["no"] = fuzz.trimf(drop.universe, [-1, -1, 0])
        drop["yes"] = fuzz.trimf(drop.universe, [0, 1, 1])

        rules = [
            ctrl.Rule(value["many"], drop["yes"]),
            ctrl.Rule(value["some"] & opp["close"], drop["yes"]),
            ctrl.Rule(value["some"] & opp["far"], drop["no"]),
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

    def time_to_impact(self, thrust, sx, sy, speed, heading, turn_rate, turn_time,
                       ast_x, ast_y, ast_vx, ast_vy, ast_r, opponents, mines, W, H):
        """How many seconds until we hit something if we hold this thrust.
        Simulates the ship (drag, speed cap, still turning toward its target)
        and checks asteroids, the other ship and mine blasts."""
        times = self.lookahead_times
        step = LOOKAHEAD_STEP

        # where the ship will be at each look-ahead time
        path_x = np.empty(len(times))
        path_y = np.empty(len(times))
        x, y, v = sx, sy, speed
        for k in range(len(times)):
            if abs(v) > 1e-9:
                if thrust == 0.0 and abs(v) <= 80.0 * step:
                    v_new = 0.0
                else:
                    v_new = v + (thrust - 80.0 * math.copysign(1.0, v)) * step
            else:
                v_new = v + (thrust - 80.0 * math.copysign(1.0, thrust)) * step if thrust != 0.0 else 0.0
            v_new = max(-240.0, min(240.0, v_new))

            # velocity always points where the ship is facing
            angle = math.radians(heading + turn_rate * min(times[k] - 0.5 * step, turn_time))
            dist = 0.5 * (v + v_new) * step
            x += dist * math.cos(angle)
            y += dist * math.sin(angle)
            path_x[k] = x
            path_y[k] = y
            v = v_new

        margin = self.params["threat_margin"]
        first_hit = NOTHING_IN_SIGHT

        if len(ast_x):
            gap_x = ast_x[None, :] + ast_vx[None, :] * times[:, None] - path_x[:, None]
            gap_y = ast_y[None, :] + ast_vy[None, :] * times[:, None] - path_y[:, None]
            gap_x = (gap_x + W / 2) % W - W / 2
            gap_y = (gap_y + H / 2) % H - H / 2
            touching = (gap_x ** 2 + gap_y ** 2) < ((ast_r + SHIP_RADIUS + margin) ** 2)[None, :]
            hit_steps = np.nonzero(touching.any(axis=1))[0]
            if len(hit_steps):
                first_hit = float(times[hit_steps[0]])

            now_x = (ast_x - sx + W / 2) % W - W / 2
            now_y = (ast_y - sy + H / 2) % H - H / 2
            if np.any(now_x ** 2 + now_y ** 2 < (ast_r + SHIP_RADIUS) ** 2):
                first_hit = 0.0

        for other in opponents:
            if other["is_respawning"]:
                continue
            ox, oy = other["position"]
            ovx, ovy = other["velocity"]
            gap_x = (ox + ovx * times - path_x + W / 2) % W - W / 2
            gap_y = (oy + ovy * times - path_y + H / 2) % H - H / 2
            hit_steps = np.nonzero(gap_x ** 2 + gap_y ** 2 < (2 * SHIP_RADIUS + margin + 10.0) ** 2)[0]
            if len(hit_steps):
                first_hit = min(first_hit, float(times[hit_steps[0]]))

        for mine in mines:
            fuse = float(mine["remaining_time"])
            k = min(len(times) - 1, max(0, int(round(fuse / step)) - 1))
            if fuse > times[-1]:
                k = len(times) - 1
            mx, my = mine["position"]
            gap_x = (mx - path_x[k] + W / 2) % W - W / 2
            gap_y = (my - path_y[k] + H / 2) % H - H / 2
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

        W, H = game_state["map_size"]
        dt = game_state["delta_time"]
        sx, sy = ship_state["position"]
        svx, svy = ship_state["velocity"]
        heading = ship_state["heading"]
        speed = ship_state["speed"]
        respawning = bool(ship_state["is_respawning"])
        respawn_left = float(ship_state["respawn_time_left"]) if respawning else 0.0

        asteroids = game_state["asteroids"]
        n = len(asteroids)
        if n:
            data = np.array([(a["position"][0], a["position"][1], a["velocity"][0], a["velocity"][1], a["radius"])
                             for a in asteroids], dtype=float)
            ast_x, ast_y, ast_vx, ast_vy, ast_r = data.T
        else:
            ast_x = ast_y = ast_vx = ast_vy = ast_r = np.zeros(0)

        opponents = [s for s in game_state["ships"] if s["id"] != ship_state["id"]]
        mines = game_state["mines"]
        bullets = game_state["bullets"]

        # How soon each asteroid hits us if we just coast. Offsets wrap around the screen.
        ttc = np.full(n, np.inf)
        if n:
            dx = (ast_x - sx + W / 2) % W - W / 2
            dy = (ast_y - sy + H / 2) % H - H / 2
            rel_vx = ast_vx - svx
            rel_vy = ast_vy - svy
            reach = ast_r + SHIP_RADIUS + params["threat_margin"]
            a = rel_vx * rel_vx + rel_vy * rel_vy
            b = 2.0 * (dx * rel_vx + dy * rel_vy)
            c = dx * dx + dy * dy - reach * reach
            disc = b * b - 4.0 * a * c
            t_hit = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
            will_hit = (disc >= 0) & (a > 1e-9) & (t_hit >= 0)
            ttc = np.where(c <= 0, 0.0, np.where(will_hit, t_hit, np.inf))

        opp_dist = 1000.0
        for other in opponents:
            ox, oy = other["position"]
            opp_dist = min(opp_dist, math.hypot((ox - sx + W / 2) % W - W / 2, (oy - sy + H / 2) % H - H / 2))

        # Asteroids that a bullet already in flight is going to destroy
        doomed = np.zeros(n, dtype=bool)
        if n and len(bullets):
            bul = np.array([(b["position"][0], b["position"][1], b["velocity"][0], b["velocity"][1])
                            for b in bullets], dtype=float)
            bx, by, bvx, bvy = bul[:, 0:1], bul[:, 1:2], bul[:, 2:3], bul[:, 3:4]
            gap_x = ast_x[None, :] - bx
            gap_y = ast_y[None, :] - by
            rel_vx = ast_vx[None, :] - bvx
            rel_vy = ast_vy[None, :] - bvy
            rel_v2 = rel_vx * rel_vx + rel_vy * rel_vy
            t_close = np.clip(-(gap_x * rel_vx + gap_y * rel_vy) / np.maximum(rel_v2, 1e-9), 0.0, 2.0)
            miss_x = gap_x + rel_vx * t_close
            miss_y = gap_y + rel_vy * t_close
            hit = (miss_x * miss_x + miss_y * miss_y) < (ast_r[None, :] + 1.0) ** 2
            # bullets disappear when they leave the map
            end_x = bx + bvx * t_close
            end_y = by + bvy * t_close
            hit &= (end_x >= 0) & (end_x <= W) & (end_y >= 0) & (end_y <= H)
            # each bullet only kills the first asteroid it reaches
            t_hit = np.where(hit, t_close, np.inf)
            first = np.argmin(t_hit, axis=1)
            got_one = np.isfinite(t_hit[np.arange(len(first)), first])
            doomed[first[got_one]] = True

        # Pick a target: the asteroid we can turn to and hit soonest.
        # Every asteroid is tried at all 9 screen-wrapped positions.
        aim_error = 0.0
        if n:
            shifts = np.array([(sx_, sy_) for sx_ in (-W, 0.0, W) for sy_ in (-H, 0.0, H)])
            tx = (ast_x[None, :] + shifts[:, 0:1]).ravel()
            ty = (ast_y[None, :] + shifts[:, 1:2]).ravel()
            tvx = np.tile(ast_vx, 9)
            tvy = np.tile(ast_vy, 9)
            tr = np.tile(ast_r, 9)
            t_doomed = np.tile(doomed, 9)
            t_ttc = np.tile(ttc, 9)

            turn_time = np.zeros_like(tx)
            for _ in range(2):  # second pass accounts for the time spent turning
                lead = turn_time + dt
                fire_x = sx + svx * lead
                fire_y = sy + svy * lead
                x0 = tx + tvx * lead - fire_x
                y0 = ty + tvy * lead - fire_y
                a = tvx * tvx + tvy * tvy - BULLET_SPEED ** 2
                b = 2.0 * (x0 * tvx + y0 * tvy) - 2.0 * BULLET_SPEED * SHIP_RADIUS
                c = x0 * x0 + y0 * y0 - SHIP_RADIUS ** 2
                disc = np.maximum(b * b - 4.0 * a * c, 0.0)
                # a == 0 only if an asteroid moves at exactly bullet speed; then it's linear
                linear = np.abs(a) < 1e-6
                safe_a = np.where(linear, -1.0, a)
                r1 = (-b + np.sqrt(disc)) / (2.0 * safe_a)
                r2 = (-b - np.sqrt(disc)) / (2.0 * safe_a)
                r_linear = np.where(np.abs(b) > 1e-9, -c / np.where(np.abs(b) > 1e-9, b, 1.0), np.inf)
                bullet_time = np.where(linear, r_linear, np.maximum(r1, r2))
                bullet_time = np.where(np.isfinite(bullet_time), np.maximum(bullet_time, 0.0), 1e9)
                hit_x = x0 + tvx * bullet_time
                hit_y = y0 + tvy * bullet_time
                angle = np.degrees(np.arctan2(hit_y, hit_x))
                diff = (angle - heading + 180.0) % 360.0 - 180.0
                turn_time = np.abs(diff) / MAX_TURN

            # the intercept has to be on the map (bullets don't wrap)
            world_x = hit_x + fire_x
            world_y = hit_y + fire_y
            valid = ((world_x > tr) & (world_x < W - tr) & (world_y > tr) & (world_y < H - tr)
                     & ~t_doomed & (bullet_time < 1.4))
            if valid.any():
                cost = turn_time + params["bullet_weight"] * bullet_time - params["threat_priority"] * np.exp(-t_ttc)
                cost = np.where(valid, cost, np.inf)
                aim_error = float(diff[int(np.argmin(cost))])

        turn_rate = fuzzy.run(fuzzy.aim, {"aim_error": float(min(max(aim_error, -30.0), 30.0))}, "turn_rate")
        turn_rate = float(min(max(turn_rate, -MAX_TURN), MAX_TURN))

        # Movement: how long until we hit something if we go forward, coast, or reverse?
        turn_time = min(abs(aim_error) / MAX_TURN, LOOKAHEAD_STEP * LOOKAHEAD_STEPS)
        t_fwd, t_coast, t_rev = (self.time_to_impact(u, sx, sy, speed, heading, turn_rate, turn_time,
                                                     ast_x, ast_y, ast_vx, ast_vy, ast_r, opponents, mines, W, H)
                                 for u in (MAX_THRUST_OUT, 0.0, -MAX_THRUST_OUT))
        danger_time = t_coast
        best = max(t_fwd, t_rev)
        escape_pref = (t_fwd - t_rev) / best if best > 1e-6 else 0.0

        mine_near = any(math.hypot((m["position"][0] - sx + W / 2) % W - W / 2,
                                   (m["position"][1] - sy + H / 2) % H - H / 2) < MINE_BLAST + SHIP_RADIUS + 30.0
                        for m in mines)

        # Just respawned and invincible: move away from nearby asteroids
        if respawning and respawn_left > 0.5 and n:
            d2 = dx * dx + dy * dy + 1.0
            near = d2 < 300.0 ** 2
            if near.any():
                away_x = -float(np.sum(dx[near] / d2[near]))
                away_y = -float(np.sum(dy[near] / d2[near]))
                length = math.hypot(away_x, away_y)
                if length > 1e-9:
                    h = math.radians(heading)
                    escape_pref = (away_x * math.cos(h) + away_y * math.sin(h)) / length
                    danger_time = min(danger_time, params["respawn_flee"])

        thrust = fuzzy.run(fuzzy.move, {"danger_time": float(min(max(danger_time, 0.0), 5.0)),
                                        "escape_pref": float(min(max(escape_pref, -1.0), 1.0)),
                                        "speed": float(min(max(speed, -240.0), 240.0))}, "thrust")

        # Shot quality: simulate the bullet we would fire this frame
        shot_quality = 3.0
        if n:
            h = math.radians(heading + turn_rate * dt)
            hx, hy = math.cos(h), math.sin(h)
            start_x = sx + svx * dt + SHIP_RADIUS * hx
            start_y = sy + svy * dt + SHIP_RADIUS * hy
            bvx, bvy = BULLET_SPEED * hx, BULLET_SPEED * hy
            gap_x = ast_x + ast_vx * dt - start_x
            gap_y = ast_y + ast_vy * dt - start_y
            rel_vx = ast_vx - bvx
            rel_vy = ast_vy - bvy
            rel_v2 = rel_vx * rel_vx + rel_vy * rel_vy
            t_close = np.clip(-(gap_x * rel_vx + gap_y * rel_vy) / np.maximum(rel_v2, 1e-9), 0.0, 1.6)
            miss_x = gap_x + rel_vx * t_close
            miss_y = gap_y + rel_vy * t_close
            ratio = np.sqrt(miss_x * miss_x + miss_y * miss_y) / ast_r
            end_x = start_x + bvx * t_close
            end_y = start_y + bvy * t_close
            usable = (end_x >= 0) & (end_x <= W) & (end_y >= 0) & (end_y <= H) & ~doomed
            hits = usable & (ratio < 1.0)
            if hits.any():
                # whichever asteroid the bullet reaches first is the one that counts
                shot_quality = float(ratio[int(np.argmin(np.where(hits, t_close, np.inf)))])
            elif usable.any():
                shot_quality = float(min(3.0, ratio[usable].min()))

        fire_out = fuzzy.run(fuzzy.fire, {"shot_quality": float(min(shot_quality, 3.0)),
                                          "respawn_left": float(min(respawn_left, 3.0))}, "fire")
        fire = bool(fire_out > 0.0)

        # Mines: only if no other mine is about to go off near us and we have a way out
        drop_mine = False
        if (ship_state["mines_remaining"] != 0 and ship_state["can_deploy_mine"] and not mine_near and n
                and max(t_fwd, t_rev) >= MINE_ESCAPE_LEAD):
            # asteroids wrap but the blast doesn't, so use plain distance here
            fx = (ast_x + ast_vx * MINE_FUSE) % W
            fy = (ast_y + ast_vy * MINE_FUSE) % H
            in_blast = ((fx - sx) ** 2 + (fy - sy) ** 2) < (MINE_BLAST + ast_r) ** 2
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
    line = "DEFAULT_CHROMOSOME = [" + ", ".join("%.3f" % g for g in chromosome) + "]"
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
