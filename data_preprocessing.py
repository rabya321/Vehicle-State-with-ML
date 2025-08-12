import os
import argparse
import pandas as pd
from datetime import datetime, timedelta

# Constants
INS_EPOCH = datetime(1980, 1, 6)
TARGET_FREQ = 50  # Hz

def convert_ins_to_datetime(week, msec):
    """Convert INS week and milliseconds to datetime."""
    return INS_EPOCH + timedelta(weeks=week, milliseconds=msec)

def load_adma(adma_path):
    """Load ADMA CSV, compute real time and elapsed time."""
    adma_df = pd.read_csv(adma_path)

    real_time_ins = adma_df.apply(
        lambda row: convert_ins_to_datetime(row["ins_time_week"], row["ins_time_msec"]),
        axis=1
    )
    unix_secs_ins = real_time_ins.apply(lambda dt: int(dt.timestamp()))
    unix_nsecs_ins = real_time_ins.apply(lambda dt: int((dt.timestamp() - int(dt.timestamp())) * 1e9))
    unix_ins = unix_secs_ins + unix_nsecs_ins * 1e-9

    adma_df["time_elapsed"] = unix_ins - unix_ins.iloc[0]
    print(f"Total simulation time: {adma_df['time_elapsed'].iloc[-1]:.2f} seconds")

    return adma_df, real_time_ins, unix_ins

def extract_vector_columns(adma_df, base_col, suffixes):
    """Extract consecutive vector columns based on a base column name."""
    start_idx = adma_df.columns.get_loc(base_col) + 1
    cols = adma_df.iloc[:, [start_idx, start_idx + 1, start_idx + 2]].copy()
    cols.columns = [f"{base_col}_{s}" for s in suffixes]
    return cols

def merge_obd_data(result_df, obd_file_path, time_prefix, columns):
    """Merge OBD CSV file into result_df using nearest timestamp."""
    df = pd.read_csv(obd_file_path)
    df[f"{time_prefix}_secs_nsecs"] = df["secs"] + df["nsecs"] * 1e-9
    return pd.merge_asof(
        result_df,
        df[[f"{time_prefix}_secs_nsecs"] + columns],
        left_on="INS time",
        right_on=f"{time_prefix}_secs_nsecs",
        direction='nearest',
        tolerance=1 / TARGET_FREQ
    )

def main(adma_path, obd_dir, output_path):
    adma_df, real_time_ins, unix_ins = load_adma(adma_path)

    # Initialize result dataframe
    result_df = pd.DataFrame({
        "timestamp_real_ins": real_time_ins,
        "INS time": unix_ins,
        "time_elapsed": adma_df["time_elapsed"]
    })

    # Required ADMA scalar columns
    req_columns = adma_df[[
        "ext_vel_an_x", "ext_vel_an_y", "ext_vel_x_corrected", "ext_vel_y_corrected",
        "inv_path_radius", "side_slip_angle", "dist_trav", "gnss_lat_abs", "gnss_long_abs",
        "gnss_pos_rel_x", "gnss_pos_rel_y", "gnss_stddev_lat", "gnss_stddev_long", "gnss_stddev_height",
        "gnss_vel_latency", "gnss_time_msec", "gnss_time_week", "gnss_sats_used", "gnss_sats_visible",
        "gnss_sats_dualant_used", "gnss_sats_dualant_visible", "ins_roll", "ins_pitch", "ins_yaw",
        "ins_height", "ins_time_msec", "ins_time_week", "leap_seconds", "ins_lat_abs", "ins_long_abs",
        "ins_pos_rel_x", "ins_pos_rel_y", "ins_stddev_roll", "ins_stddev_pitch", "ins_stddev_yaw",
        "an1", "an2", "an3", "an4", "kf_lat_stimulated", "kf_long_stimulated", "kf_steady_state",
        "status_gnss_mode", "status_tilt", "status_pos", "status_kalmanfilter_settled",
        "status_kf_lat_stimulated", "status_kf_long_stimulated", "status_kf_steady_state", "status_speed"
    ]]
    result_df = pd.concat([result_df, req_columns], axis=1)

    # ADMA vector columns
    result_df = pd.concat([result_df, extract_vector_columns(adma_df, "rate_hor", ["x", "y", "z"])], axis=1)
    result_df = pd.concat([result_df, extract_vector_columns(adma_df, "acc_hor", ["x", "y", "z"])], axis=1)
    result_df = pd.concat([result_df, extract_vector_columns(adma_df, "gnss_vel_frame", ["x", "y", "z"])], axis=1)
    result_df = pd.concat([result_df, extract_vector_columns(adma_df, "gnss_stddev_vel", ["x", "y", "z"])], axis=1)
    result_df = pd.concat([result_df, extract_vector_columns(adma_df, "ins_vel_hor", ["x", "y", "z"])], axis=1)
    result_df = pd.concat([result_df, extract_vector_columns(adma_df, "ins_stddev_vel", ["x", "y", "z"])], axis=1)

    # Merge OBD data files
    obd_files = [
        ("_slash_BrakePressInfo.csv", "brake", ["BrakePressInDec", "BrakeLight"]),
        ("_slash_LatAccelInfo.csv", "lat", ["LateralAcceleration"]),
        ("_slash_PedalAndDirectionInfo.csv", "pad", ["FahrtrichtungForward", "FahrtrichtungBackwards",
                                                     "FahrtrichtungUnknown", "Bremslichtschalter", "MEGCRG"]),
        ("_slash_SpeedoInfo.csv", "speedo", ["SpeedoInDec"]),
        ("_slash_SteerWheelInfo.csv", "steer", ["SWPosinDec"]),
        ("_slash_VelFrontInfo.csv", "velf", ["VelFRInDec", "VelFLInDec"]),
        ("_slash_VelRearInfo.csv", "velr", ["VelRRInDec", "VelRLInDec"]),
        ("_slash_YawRateInfo.csv", "yaw", ["GierrateDegXSec"])
    ]

    for filename, prefix, cols in obd_files:
        file_path = os.path.join(obd_dir, filename)
        if os.path.exists(file_path):
            result_df = merge_obd_data(result_df, file_path, prefix, cols)
        else:
            print(f"Warning: {filename} not found in {obd_dir}")

    # Save synchronized data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved synchronized data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronize ADMA and OBD-II data to 50 Hz.")
    parser.add_argument("--adma", required=True, help="Path to ADMA CSV file")
    parser.add_argument("--obd", required=True, help="Directory containing OBD CSV files")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    main(args.adma, args.obd, args.output)
    
# This script synchronizes ADMA and OBD data to a common 50 Hz frequency.