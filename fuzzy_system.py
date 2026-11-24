
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ==========================================
# INPUT VARIABLES
# ==========================================

# 1. Distance (metres)
distance = ctrl.Antecedent(np.arange(0, 1501, 1), 'distance')

distance['near'] = fuzz.trapmf(
    distance.universe, [0, 0, 150, 350]
)
distance['medium'] = fuzz.trimf(
    distance.universe, [200, 500, 800]
)
distance['far'] = fuzz.trapmf(
    distance.universe, [650, 1000, 1500, 1500]
)


# 2. Aim error (degrees)
aim_error = ctrl.Antecedent(np.arange(-180, 181, 1), 'aim_error')

aim_error['far_left'] = fuzz.trapmf(
    aim_error.universe, [-180, -180, -60, -20]
)

aim_error['slight_left'] = fuzz.trimf(
    aim_error.universe, [-45, -15, 0]
)

aim_error['center'] = fuzz.trimf(
    aim_error.universe, [-10, 0, 10]
)

aim_error['slight_right'] = fuzz.trimf(
    aim_error.universe, [0, 15, 45]
)

aim_error['far_right'] = fuzz.trapmf(
    aim_error.universe, [20, 60, 180, 180]
)


# 3. Collision risk (0-100)
collision_risk = ctrl.Antecedent(np.arange(0, 101, 1), 'collision_risk')

collision_risk['low'] = fuzz.trapmf(collision_risk.universe, [0, 0, 20, 40])
collision_risk['medium'] = fuzz.trimf(collision_risk.universe, [25, 50, 75])
collision_risk['high'] = fuzz.trapmf(collision_risk.universe, [60, 80, 100, 100])


# 4. Relative velocity (m/s)
# Positive = approaching, negative = receding
relative_velocity = ctrl.Antecedent(
    np.arange(-1001, 1001, 1), 'relative_velocity'
)

relative_velocity['receding'] = fuzz.trapmf(
    relative_velocity.universe, [-1000, -1000, -200, -50]
)

relative_velocity['stable'] = fuzz.trimf(
    relative_velocity.universe, [-100, 0, 100]
)

relative_velocity['slow_approaching'] = fuzz.trimf(
    relative_velocity.universe, [50, 200, 400]
)

relative_velocity['fast_approaching'] = fuzz.trapmf(
    relative_velocity.universe, [250, 500, 1000, 1000]
)

# 5. Asteroid radius (metres)
# Asteroid radius (metres)
asteroid_size = ctrl.Antecedent(np.arange(0, 33, 1), 'asteroid_size')

asteroid_size['small'] = fuzz.trapmf(
    asteroid_size.universe, [0, 0, 8, 16]
)

asteroid_size['medium'] = fuzz.trimf(
    asteroid_size.universe, [8, 16, 24]
)

asteroid_size['large'] = fuzz.trapmf(
    asteroid_size.universe, [16, 24, 32, 32]
)


# 6. Time to intercept (seconds)
time_to_intercept = ctrl.Antecedent(
    np.arange(0, 10.1, 0.1), 'time_to_intercept'
)

time_to_intercept['short'] = fuzz.trapmf(
    time_to_intercept.universe, [0, 0, 1, 3]
)
time_to_intercept['medium'] = fuzz.trimf(
    time_to_intercept.universe, [2, 5, 8]
)
time_to_intercept['long'] = fuzz.trapmf(
    time_to_intercept.universe, [6, 8, 10, 10]
)


# ==========================================
# OUTPUT VARIABLES
# ==========================================

# 1. Turn rate (degrees/second)
turn_rate = ctrl.Consequent(np.arange(-180, 181, 1), 'turn_rate')

turn_rate['hard_left'] = fuzz.trapmf(
    turn_rate.universe, [-180, -180, -120, -60]
)

turn_rate['soft_left'] = fuzz.trimf(
    turn_rate.universe, [-100, -40, 0]
)

turn_rate['straight'] = fuzz.trimf(
    turn_rate.universe, [-15, 0, 15]
)

turn_rate['soft_right'] = fuzz.trimf(
    turn_rate.universe, [0, 40, 100]
)

turn_rate['hard_right'] = fuzz.trapmf(
    turn_rate.universe, [60, 120, 180, 180]
)


# 2. Thrust (normalized 0-1)
thrust = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'thrust')

thrust['low'] = fuzz.trapmf(
    thrust.universe, [0, 0, 0.2, 0.4]
)
thrust['medium'] = fuzz.trimf(
    thrust.universe, [0.2, 0.5, 0.8]
)
thrust['high'] = fuzz.trapmf(
    thrust.universe, [0.6, 0.8, 1, 1]
)


# 3. Fire (normalized 0-1)
fire = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'fire')

fire['no'] = fuzz.trapmf(
    fire.universe, [0, 0, 0.3, 0.5]
)
fire['yes'] = fuzz.trapmf(
    fire.universe, [0.4, 0.7, 1, 1]
)

# 4. Mine deployment (normalized 0-1)
drop_mine = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'drop_mine')

drop_mine['no'] = fuzz.trapmf(
    drop_mine.universe, [0, 0, 0.3, 0.5]
)

drop_mine['yes'] = fuzz.trapmf(
    drop_mine.universe, [0.4, 0.7, 1, 1]
)


# ==========================================
# FUZZY RULES
# ==========================================

rules = [

    # TURN RATE RULES
    ctrl.Rule(aim_error['far_left'], turn_rate['hard_left']),
    ctrl.Rule(aim_error['slight_left'], turn_rate['soft_left']),
    ctrl.Rule(aim_error['center'], turn_rate['straight']),
    ctrl.Rule(aim_error['slight_right'], turn_rate['soft_right']),
    ctrl.Rule(aim_error['far_right'], turn_rate['hard_right']),


    # THRUST RULES

    # High collision risk: reduce thrust
    ctrl.Rule(collision_risk['high'], thrust['low']),

    # Medium collision risk: moderate thrust
    ctrl.Rule(collision_risk['medium'], thrust['medium']),

    # Low collision risk: adjust thrust based on distance
    ctrl.Rule(
        collision_risk['low'] & distance['far'],
        thrust['high']
    ),

    ctrl.Rule(
        collision_risk['low'] & distance['medium'],
        thrust['medium']
    ),

    # Nearby asteroids: adjust thrust based on relative velocity
    ctrl.Rule(
        collision_risk['low'] &
        distance['near'] &
        relative_velocity['fast_approaching'],
        thrust['low']
    ),

    ctrl.Rule(
        collision_risk['low'] &
        distance['near'] &
        relative_velocity['slow_approaching'],
        thrust['low']
    ),

    ctrl.Rule(
        collision_risk['low'] &
        distance['near'] &
        relative_velocity['stable'],
        thrust['low']
    ),

    ctrl.Rule(
        collision_risk['low'] &
        distance['near'] &
        relative_velocity['receding'],
        thrust['medium']
    ),


    # FIRE RULES
    ctrl.Rule(aim_error['far_left'], fire['no']),
    ctrl.Rule(aim_error['slight_left'], fire['no']),
    ctrl.Rule(aim_error['slight_right'], fire['no']),
    ctrl.Rule(aim_error['far_right'], fire['no']),
    ctrl.Rule(distance['far'], fire['no']),
    ctrl.Rule(time_to_intercept['long'], fire['no']),
    ctrl.Rule(asteroid_size['small'], fire['no']),
    ctrl.Rule(collision_risk['high'], fire['no']),

    ctrl.Rule(
        aim_error['center'] &
        (distance['near'] | distance['medium']) &
        (time_to_intercept['short'] | time_to_intercept['medium']) &
        (asteroid_size['medium'] | asteroid_size['large']) &
        (collision_risk['low'] | collision_risk['medium']),
        fire['yes']
    ),

    # MINE DEPLOYMENT RULES

    # Do not deploy when collision risk is low
    ctrl.Rule(collision_risk['low'], drop_mine['no']),

    # Do not deploy when the asteroid is far away
    ctrl.Rule(distance['far'], drop_mine['no']),

    # Do not deploy for small asteroids
    ctrl.Rule(asteroid_size['small'], drop_mine['no']),

    # Do not deploy when the asteroid is stable or receding
    ctrl.Rule(
        relative_velocity['stable'] | relative_velocity['receding'],
        drop_mine['no']
    ),

    # Deploy when a large or medium asteroid is nearby,
    # collision risk is high, and the asteroid is approaching
    ctrl.Rule(
        distance['near'] &
        collision_risk['high'] &
        (asteroid_size['medium'] | asteroid_size['large']) &
        (relative_velocity['fast_approaching'] |
        relative_velocity['slow_approaching']),
        drop_mine['yes']
    ),
]


# ==========================================
# FUZZY CONTROL SYSTEM
# ==========================================

fuzzy_system = ctrl.ControlSystem(rules)
fuzzy_simulation = ctrl.ControlSystemSimulation(fuzzy_system)


# ==========================================
# CALCULATION FUNCTION
# ==========================================

def calculate(distance_value, aim_error_value, collision_risk_value,
              relative_velocity_value, asteroid_size_value,
              time_to_intercept_value):

    # Keep inputs within their defined universes
    inputs = {
        'distance': (distance_value, 0, 1500),
        'aim_error': (aim_error_value, -180, 180),
        'collision_risk': (collision_risk_value, 0, 100),
        'relative_velocity': (relative_velocity_value, -1000, 1000),
        'asteroid_size': (asteroid_size_value, 0, 32),
        'time_to_intercept': (time_to_intercept_value, 0, 10)
    }

    for name, (value, minimum, maximum) in inputs.items():
        fuzzy_simulation.input[name] = np.clip(value, minimum, maximum)

    fuzzy_simulation.compute()

    turn = fuzzy_simulation.output['turn_rate']
    thrust_value = fuzzy_simulation.output['thrust']
    fire_value = fuzzy_simulation.output['fire']
    mine_value = fuzzy_simulation.output['drop_mine']

    return (
        turn,
        thrust_value,
        fire_value >= 0.5,
        mine_value >= 0.5
    )


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":

    tests = [
        ("Nearby asteroid", 100, 0, 10, 50, 32, 2),
        ("Medium distance", 500, 0, 10, 0, 16, 5),
        ("Far asteroid", 1200, 0, 10, -50, 32, 8),

        ("Fast approaching", 100, 0, 10, 300, 32, 1.5),
        ("Receding asteroid", 100, 0, 10, -300, 32, 3),

        ("High collision risk", 100, 0, 90, 300, 32, 1),

        ("Good firing opportunity", 200, 0, 10, 0, 32, 2),
        ("Asteroid too far", 1200, 0, 10, 0, 32, 8),
        ("Asteroid too small", 200, 0, 10, 0, 8, 2),
        ("Intercept too long", 200, 0, 10, 0, 32, 9),
        ("Aim slightly off", 200, 20, 10, 0, 32, 2),

        ("Mine deployment opportunity", 100, 0, 90, 300, 32, 1.5),
    ]

    for name, d, angle, risk, velocity, size, intercept in tests:

        result = calculate(
            distance_value=d,
            aim_error_value=angle,
            collision_risk_value=risk,
            relative_velocity_value=velocity,
            asteroid_size_value=size,
            time_to_intercept_value=intercept
        )

        print(f"\n{name}")
        print("Turn rate:", result[0])
        print("Thrust:", result[1])
        print("Fire:", result[2])
        print("Drop mine:", result[3])      







