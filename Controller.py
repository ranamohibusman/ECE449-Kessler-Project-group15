# Controller for ECE 449 Kessler Project Group15



from kesslergame import KesslerController # In Eclipse, the name of the library is
kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt

#OR WHATEVER IT ENDS UP BEING FOR THE FINAL IMPLEMENTATION, JUST GET CALCULATE IN HERE 
from fuzzy_system import calculate

#things set in the kessler game
BULLET_SPEED = 800.0
MAX_THRUST = 480.0
DT = 1/30

def wrap_deg(a): #convert 360 degree range to [-180, 180]
  return (a + 180) % 360 - 180

def intercept_time(rx, ry, vx, vy, speed = BULLET_SPEED):
  a = vx*vx + vy*vy - speed*speed
  b = 2 * (rx*vx + ry*vy)
  c = rx*rx + ry*ry
  disc = b*b - 4*a*c

  if disc < 0 or abs(a) < 1e-9:
    return None
  sq =  math.sqrt(disc)
  positive = [t for t in ((-b - sq)/(2*a), (-b + sq)/(2*a)) if t >= 0]
  return min(positive) if positive else None


def collision_risk(rx,ry,vx,vy,hit_radius):
  vv = vx*vx + vy*vy
  if vv < 1e-9:
    return 0.0
  t_closest = -(rx*vx + ry*vy) / vv
  if t_closest <= 0:
    return 0.0
  miss = math.hypot(rx + vx*t_closest, ry + vy*t_closest)
  if miss > hit_radius * 1.5:
    return 0.0
  return 100.0 * max(0.0, 1.0 - t_closest / 5.0) #5s breathing room, something to consider for later 




class AsteroidAnnihilator(KesslerController):
  def __init__(self):
    self.eval_frames = 0

    # the fuzzy logic either is called from or exists here as far as im aware
  
  def actions(self, ship_state: Dict, game_state: Dict):
    self.eval_frames += 1

    asteroids = game_state["asteroids"]
    #asteroids in game state gives us position (x, y), velocity (vx, vy), size (integer tier, typically 1–4), radius, mass, and id

    sx, sy = ship_state["position"]
    svx, svy = ship_state["velocity"]
    heading = ship_state["heading"] #heading is given in degrees
    ship_r = ship_state.get("radius", 20)


    #find the nearest asteroid, and the worst collision risk of any current asteroid
    target, best_dist, risk = None, float("inf"), 0.0
    for a in asteroids:
      rx, ry = a["position"][0] - sx, a["position"][1] - sy
      vx, vy = a["velocity"][0] - svx, a["velocity"][1] - svy
      risk = max(risk, collision_risk(rx, ry, vx, vy, a["radius"] + ship_r))
      d = math.hypot(rx, ry)
      if d < best_dist:
        target, best_dist = (a, rx, ry, vx, vy), d
    a, rx, ry, vx, vy = target
    dist = max(best_dist, 1e6) #obv is dist is 1e6 theres nothing to be worried about


    #closing speed, positive = approaching
    closing = -(rx*vx + ry*vy) / dist


    #bullet intercept time and lead angle
    t = intercept_time(rx,ry,vx,vy)
    if t is None:
      t = dist/ BULLET_SPEED
    aim_x = rx + vx * (t + DT)
    aim_y = ry + vy * (t + DT)
    aim_error = wrap_deg(math.degrees(math.atan2(aim_y,aim_x)) - heading)


    try:
      turn, thrust, fire, mine = calculate(
          distance_value = dist,
          aim_error_value = aim_error,
          collision_risk_value = closing,
          asteroid_size_value = a["radius"],
          time_to_intercept_value = t
      )
    except ValueError:#hopefully no value error but if no rule is fired upon
      return 0.0, max(-180, min(180, aim_error * 5)), False, False #maaaybe if theres an error just start blastin?
    
    return thrust * MAX_THRUST, float(turn), bool(fire), bool(mine)

@property
def name(self) -> str:
  return "AsteroidAnnihilator"
