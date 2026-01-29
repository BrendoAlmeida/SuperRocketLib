from .core.structures import (
    Choice,
    IntRange,
    Range,
    MotorRanges,
    NoseConeRanges,
    ParachuteRanges,
    RailButtonsRanges,
    RocketConfigRanges,
    RocketRanges,
    EnvironmentRanges,
    FlightRanges,
    TailRanges,
    TrapezoidalFinsRanges,
)

SMALL_ROCKET_RANGES = RocketRanges(
    radius=Range.continuous(0.02, 0.05),
    length=Range.continuous(0.6, 1.2),
    mass=Range.continuous(0.5, 2.0),
    center_of_mass_without_motor=Range.continuous(0.2, 0.8),
    I_11_without_motor=Range.continuous(0.001, 0.01),
    I_22_without_motor=Range.continuous(0.001, 0.01),
    I_33_without_motor=Range.continuous(0.001, 0.01),
    I_12_without_motor=Range.continuous(-0.001, 0.001),
    I_13_without_motor=Range.continuous(-0.001, 0.001),
    I_23_without_motor=Range.continuous(-0.001, 0.001),
    wall_thickness=Range.continuous(0.001, 0.003),
    motor_position=Range.continuous(0.05, 0.2),
)

SMALL_MOTOR_RANGES = MotorRanges(
    motor_diameter=Range.continuous(0.018, 0.04),
    motor_length=Range.continuous(0.2, 0.45),
    nozzle_radius=Range.continuous(0.004, 0.01),
    throat_radius=Range.continuous(0.002, 0.006),
    dry_mass=Range.continuous(0.15, 0.6),
    dry_I_11=Range.continuous(0.0001, 0.001),
    dry_I_22=Range.continuous(0.0001, 0.001),
    dry_I_33=Range.continuous(0.00005, 0.0005),
    dry_I_12=Range.continuous(-0.0001, 0.0001),
    dry_I_13=Range.continuous(-0.0001, 0.0001),
    dry_I_23=Range.continuous(-0.0001, 0.0001),
    grain_number=IntRange(1, 4),
    grain_density=Range.continuous(1500, 1800),
    grain_outer_radius=Range.continuous(0.006, 0.018),
    grain_initial_inner_radius=Range.continuous(0.002, 0.01),
    grain_initial_height=Range.continuous(0.04, 0.12),
    grain_separation=Range.continuous(0.001, 0.01),
    grains_center_of_mass_position=Range.continuous(0.05, 0.2),
    center_of_dry_mass_position=Range.continuous(0.05, 0.2),
    burn_time=Range.continuous(0.6, 2.0),
    average_thrust=Range.continuous(80, 500),
    peak_thrust_ratio=Range.continuous(1.5, 2.5),
    ignition_duration_fraction=Range.continuous(0.03, 0.08),
    tail_off_fraction=Range.continuous(0.05, 0.12),
    main_burn_end_ratio=Range.continuous(0.8, 1.2),
    thrust_profile_type=Choice(["progressive", "neutral", "regressive"]),
    thrust_curve_points=IntRange(120, 200),
)

SMALL_NOSECONE_RANGES = NoseConeRanges(
    length=Range.continuous(0.12, 0.3),
    kind=Choice(["conical", "ogive", "von karman", "parabolic"]),
    bluffness=Range.continuous(0.0, 0.2),
    power=Range.continuous(0.3, 1.0),
)

SMALL_FINS_RANGES = TrapezoidalFinsRanges(
    n=IntRange(3, 4),
    root_chord=Range.continuous(0.08, 0.16),
    tip_chord=Range.continuous(0.03, 0.1),
    span=Range.continuous(0.04, 0.1),
    sweep_length=Range.continuous(0.01, 0.05),
    cant_angle=Range.continuous(0.0, 2.0),
)

SMALL_TAIL_RANGES = TailRanges(
    length=Range.continuous(0.05, 0.12),
    top_radius_ratio=Range.continuous(0.95, 1.0),
    bottom_radius_ratio=Range.continuous(0.7, 1.0),
)

SMALL_PARACHUTE_RANGES = ParachuteRanges(
    cd_s=Range.continuous(0.5, 2.0),
    trigger_altitude=Range.continuous(50, 200),
    sampling_rate=Range.continuous(50, 200),
    lag=Range.continuous(0.2, 1.0),
    noise_mean=Range.continuous(0.0, 0.0),
    noise_std=Range.continuous(0.0, 1.0),
    noise_time_correlation=Range.continuous(0.0, 0.1),
)

SMALL_RAIL_BUTTONS_RANGES = RailButtonsRanges(
    upper_button_position=Range.continuous(0.6, 1.0),
    lower_button_position=Range.continuous(0.2, 0.4),
    angular_position=Range.continuous(30, 60),
)

SMALL_ENVIRONMENT_RANGES = EnvironmentRanges(
    gravity=Range.continuous(9.78, 9.83),
    latitude=Range.continuous(-30.0, -10.0),
    longitude=Range.continuous(-60.0, -40.0),
    elevation=Range.continuous(0.0, 200.0),
    max_expected_height=Range.continuous(1000.0, 10000.0),
)

SMALL_FLIGHT_RANGES = FlightRanges(
    rail_length=Range.continuous(1.0, 3.0),
    inclination=Range.continuous(80.0, 90.0),
    heading=Range.continuous(0.0, 360.0),
    max_time=Range.continuous(120.0, 600.0),
)

SMALL_CONFIG = RocketConfigRanges(
    rocket=SMALL_ROCKET_RANGES,
    motor=SMALL_MOTOR_RANGES,
    nosecone=SMALL_NOSECONE_RANGES,
    fins=SMALL_FINS_RANGES,
    tail=SMALL_TAIL_RANGES,
    parachute=SMALL_PARACHUTE_RANGES,
    rail_buttons=SMALL_RAIL_BUTTONS_RANGES,
    environment=SMALL_ENVIRONMENT_RANGES,
    flight=SMALL_FLIGHT_RANGES,
)

STANDARD_COMPETITION_ROCKET_RANGES = RocketRanges(
    radius=Range.continuous(0.05, 0.08), 
    length=Range.continuous(2.0, 3.5),
    mass=Range.continuous(8.0, 18.0),
    center_of_mass_without_motor=Range.continuous(1.0, 2.0),
    I_11_without_motor=Range.continuous(0.1, 1.0),
    I_22_without_motor=Range.continuous(0.1, 1.0),
    I_33_without_motor=Range.continuous(0.01, 0.1),
    I_12_without_motor=Range.continuous(-0.05, 0.05),
    I_13_without_motor=Range.continuous(-0.05, 0.05),
    I_23_without_motor=Range.continuous(-0.05, 0.05),
    
    wall_thickness=Range.continuous(0.002, 0.005),
    motor_position=Range.continuous(0.3, 0.8),
)

STANDARD_COMPETITION_MOTOR_RANGES = MotorRanges(
    motor_diameter=Range.continuous(0.075, 0.098), 
    motor_length=Range.continuous(0.5, 1.2),
    nozzle_radius=Range.continuous(0.015, 0.03),
    throat_radius=Range.continuous(0.008, 0.015),
    dry_mass=Range.continuous(1.5, 4.0),
    dry_I_11=Range.continuous(0.01, 0.1),
    dry_I_22=Range.continuous(0.01, 0.1),
    dry_I_33=Range.continuous(0.001, 0.01),
    dry_I_12=Range.continuous(-0.01, 0.01),
    dry_I_13=Range.continuous(-0.01, 0.01),
    dry_I_23=Range.continuous(-0.01, 0.01),
    grain_number=IntRange(3, 5),
    grain_density=Range.continuous(1600, 1850),
    grain_outer_radius=Range.continuous(0.03, 0.045), 
    grain_initial_inner_radius=Range.continuous(0.01, 0.02),
    grain_initial_height=Range.continuous(0.1, 0.25),
    grain_separation=Range.continuous(0.005, 0.015),
    grains_center_of_mass_position=Range.continuous(0.2, 0.6),
    center_of_dry_mass_position=Range.continuous(0.2, 0.6),
    burn_time=Range.continuous(2.0, 5.0),
    average_thrust=Range.continuous(1000, 3500), 
    peak_thrust_ratio=Range.continuous(1.4, 2.2),
    ignition_duration_fraction=Range.continuous(0.03, 0.07),
    tail_off_fraction=Range.continuous(0.05, 0.12),
    main_burn_end_ratio=Range.continuous(0.8, 1.2),
    thrust_profile_type=Choice(["progressive", "neutral", "regressive"]),
    thrust_curve_points=IntRange(140, 220),
)

STANDARD_COMPETITION_NOSECONE_RANGES = NoseConeRanges(
    length=Range.continuous(0.6, 1.2),
    kind=Choice(["conical", "ogive", "von karman", "parabolic"]),
    bluffness=Range.continuous(0.0, 0.15),
    power=Range.continuous(0.3, 1.0),
)

STANDARD_COMPETITION_FINS_RANGES = TrapezoidalFinsRanges(
    n=IntRange(3, 4),
    root_chord=Range.continuous(0.10, 0.25), 
    tip_chord=Range.continuous(0.05, 0.15),
    span=Range.continuous(0.08, 0.18),
    sweep_length=Range.continuous(0.02, 0.1),
    cant_angle=Range.continuous(0.0, 1.0),
)

STANDARD_COMPETITION_TAIL_RANGES = TailRanges(
    length=Range.continuous(0.2, 0.6),
    top_radius_ratio=Range.continuous(0.95, 1.0),
    bottom_radius_ratio=Range.continuous(0.7, 1.0),
)

STANDARD_COMPETITION_PARACHUTE_RANGES = ParachuteRanges(
    cd_s=Range.continuous(8.0, 20.0),
    trigger_altitude=Range.continuous(300, 800),
    sampling_rate=Range.continuous(50, 200),
    lag=Range.continuous(0.4, 1.5),
    noise_mean=Range.continuous(0.0, 0.0),
    noise_std=Range.continuous(0.0, 1.0),
    noise_time_correlation=Range.continuous(0.0, 0.1),
)

STANDARD_COMPETITION_RAIL_BUTTONS_RANGES = RailButtonsRanges(
    upper_button_position=Range.continuous(1.6, 2.6),
    lower_button_position=Range.continuous(0.6, 1.2),
    angular_position=Range.continuous(30, 60),
)

STANDARD_COMPETITION_ENVIRONMENT_RANGES = EnvironmentRanges(
    gravity=Range.continuous(9.78, 9.83),
    latitude=Range.continuous(-30.0, -10.0),
    longitude=Range.continuous(-60.0, -40.0),
    elevation=Range.continuous(0.0, 1000.0), 
    max_expected_height=Range.continuous(2000.0, 15000.0),
)

STANDARD_COMPETITION_FLIGHT_RANGES = FlightRanges(
    rail_length=Range.continuous(3.0, 8.0),
    inclination=Range.continuous(80.0, 90.0),
    heading=Range.continuous(0.0, 360.0),
    max_time=Range.continuous(200.0, 800.0),
)

STANDARD_COMPETITION_CONFIG = RocketConfigRanges(
    rocket=STANDARD_COMPETITION_ROCKET_RANGES,
    motor=STANDARD_COMPETITION_MOTOR_RANGES,
    nosecone=STANDARD_COMPETITION_NOSECONE_RANGES,
    fins=STANDARD_COMPETITION_FINS_RANGES,
    tail=STANDARD_COMPETITION_TAIL_RANGES,
    parachute=STANDARD_COMPETITION_PARACHUTE_RANGES,
    rail_buttons=STANDARD_COMPETITION_RAIL_BUTTONS_RANGES,
    environment=STANDARD_COMPETITION_ENVIRONMENT_RANGES,
    flight=STANDARD_COMPETITION_FLIGHT_RANGES,
)

DEFAULT_CONFIG = STANDARD_COMPETITION_CONFIG