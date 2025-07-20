"""
Schema definitions for horse racing data processing.

This module defines column data types for casting raw XML data extracted from
Equibase horse racing files. Used by various files to ensure
consistent data types across the processing pipeline.

The COLUMN_TYPES dictionary maps column names to their appropriate Polars data types,
enabling efficient batch casting of 70+ columns during DataFrame creation or transformation.
"""

import polars as pl

COLUMN_TYPES = {
    # "race_date": pl.Utf8, # Will be manually casted to datetime.
    "track_code": pl.Utf8,
    "track_name": pl.Utf8,
    "race_type": pl.Utf8,
    "race_number": pl.Int64,
    "race_purse": pl.Int64,
    "distance": pl.Int64,
    "distance_unit": pl.Utf8,
    "course_id": pl.Utf8,
    "course_desc": pl.Utf8,
    "course_surface": pl.Utf8,
    "class_rating": pl.Int64,
    "track_conditions": pl.Utf8,
    "weather": pl.Utf8,
    "start_desc": pl.Utf8,
    "runup_distance": pl.Int64,
    "rail_distance": pl.Int64,
    "sealed": pl.Utf8,
    "fraction_1": pl.Float64,
    "fraction_2": pl.Float64,
    "fraction_3": pl.Float64,
    "fraction_4": pl.Float64,
    "fraction_5": pl.Float64,
    "win_time": pl.Float64,
    "pace_call_1": pl.Int64,
    "pace_call_2": pl.Int64,
    "pace_call_final": pl.Int64,
    "par_time": pl.Float64,
    "footnotes": pl.Utf8,
    "horse_name": pl.Utf8,
    "jockey_first_name": pl.Utf8,
    "jockey_last_name": pl.Utf8,
    "jockey_weight": pl.Int64,
    "horse_age": pl.Int64,
    "sex_code": pl.Utf8,
    "sex_desc": pl.Utf8,
    "last_pp_track_code": pl.Utf8,
    "last_pp_track_name": pl.Utf8,
    # "last_pp_race_date": pl.Utf8, # Will be manually casted to datetime.
    "last_pp_race_number": pl.Int64,
    "last_pp_race_finish": pl.Int64,
    "meds": pl.Utf8,
    "equip": pl.Utf8,
    "dollar_odds": pl.Float64,
    "program_number": pl.Utf8,  # Cast as strings due to results such as "1A"
    "post_position": pl.Int64,
    "start_position": pl.Int64,
    "point_of_call_1_position": pl.Int64,
    "point_of_call_1_lengths": pl.Float64,
    "point_of_call_2_position": pl.Int64,
    "point_of_call_2_lengths": pl.Float64,
    "point_of_call_3_position": pl.Int64,
    "point_of_call_3_lengths": pl.Float64,
    "point_of_call_4_position": pl.Int64,
    "point_of_call_4_lengths": pl.Float64,
    "point_of_call_5_position": pl.Int64,
    "point_of_call_5_lengths": pl.Float64,
    "point_of_call_final_position": pl.Int64,
    "point_of_call_final_lengths": pl.Float64,
    "official_final_position": pl.Int64,
    "speed_rating": pl.Int64,
    "jockey_type": pl.Utf8,
    "trainer_first_name": pl.Utf8,
    "trainer_last_name": pl.Utf8,
    "trainer_type": pl.Utf8,
    "owner_full_name": pl.Utf8,
    "comment": pl.Utf8,
}
