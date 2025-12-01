"""Estimate actuator delay and PD gains from CAN bus data.

Fits a delayed PD model to measured motor torques:
    tau_measured = Kp * (cmd_pos - fb_pos) + Kd * (-fb_vel)

For each candidate delay d, fits Kp/Kd via least-squares, then picks
the delay that gives the lowest MSE between predicted and measured torque.

The identified Kp/Kd minimize sim-to-real torque error.

Usage:
    python scripts/estimate_actuator_delay.py --csv data.csv
    python scripts/estimate_actuator_delay.py --csv data.csv --kd-cap 5.0
"""

import argparse

import numpy as np
import pandas as pd

# Motor torque constants (Nm/A)
MOTOR_KT = {
    1: 2.8, 2: 7.26, 3: 5.7, 4: 2.53, 5: 1.8, 6: 1.8,
    7: 2.8, 8: 7.26, 9: 5.7, 10: 2.53, 11: 1.8, 12: 1.8,
}

JOINT_NAMES = {
    1: "L_Hip_Pitch", 2: "L_Hip_Roll", 3: "L_Hip_Yaw", 4: "L_Knee",
    5: "L_Ankle_Pitch", 6: "L_Ankle_Roll",
    7: "R_Hip_Pitch", 8: "R_Hip_Roll", 9: "R_Hip_Yaw", 10: "R_Knee",
    11: "R_Ankle_Pitch", 12: "R_Ankle_Roll",
}

JOINT_TYPE = {
    1: "hip_pitch", 2: "hip_roll", 3: "hip_yaw", 4: "knee", 5: "ankle_pitch", 6: "ankle_roll",
    7: "hip_pitch", 8: "hip_roll", 9: "hip_yaw", 10: "knee", 11: "ankle_pitch", 12: "ankle_roll",
}

# Firmware Kp/Kd for comparison (from motor_map.h)
FIRMWARE_KP = {
    1: 40.18, 2: 99.11, 3: 40.18, 4: 99.11, 5: 14.25, 6: 14.25,
    7: 40.18, 8: 99.11, 9: 40.18, 10: 99.11, 11: 14.25, 12: 14.25,
}
FIRMWARE_KD = {
    1: 2.56, 2: 5.0, 3: 2.56, 4: 5.0, 5: 0.91, 6: 0.91,
    7: 2.56, 8: 5.0, 9: 2.56, 10: 5.0, 11: 0.91, 12: 0.91,
}


def fit_pd_gains_for_motor(
    motor_id: int,
    cmd_pos: np.ndarray,
    fb_pos: np.ndarray,
    fb_vel: np.ndarray,
    fb_cur: np.ndarray,
    max_delay: int = 20,
    kd_cap: float | None = None,
) -> dict:
    """Fit Kp, Kd, and delay for a single motor via least-squares.

    Model: tau_measured = Kp * (cmd_pos - fb_pos) + Kd * (-fb_vel)

    For each delay d, fits Kp/Kd via least-squares, picks d with lowest MSE.

    Args:
        motor_id: Motor ID (1-12)
        cmd_pos: Commanded position array
        fb_pos: Feedback position array
        fb_vel: Feedback velocity array
        fb_cur: Feedback current array
        max_delay: Maximum delay in samples to search
        kd_cap: Optional cap on Kd (for motor safety)

    Returns:
        Dict with delay_samples, Kp, Kd, rmse, etc.
    """
    K_t = MOTOR_KT[motor_id]

    # Compute signals
    e_pos = cmd_pos - fb_pos          # position error
    e_vel = -fb_vel                   # damping term (Kd * -velocity)
    tau_meas = K_t * fb_cur           # measured torque

    best_mse = float("inf")
    best_result = None

    for d in range(max_delay + 1):
        if d >= len(tau_meas) - 5:
            break

        # Shift error signals by delay d
        if d == 0:
            X = np.column_stack([e_pos, e_vel])
            y = tau_meas
        else:
            X = np.column_stack([e_pos[:-d], e_vel[:-d]])
            y = tau_meas[d:]

        n = min(len(X), len(y))
        Xn, yn = X[:n], y[:n]

        # Least-squares fit: [Kp, Kd]
        K, *_ = np.linalg.lstsq(Xn, yn, rcond=None)
        Kp_fit, Kd_fit = K[0], K[1]

        # Apply Kd cap if specified
        if kd_cap is not None and Kd_fit > kd_cap:
            Kd_fit = kd_cap

        # Recompute prediction with (possibly capped) gains
        y_pred = Kp_fit * Xn[:, 0] + Kd_fit * Xn[:, 1]
        mse = np.mean((yn - y_pred) ** 2)
        rmse = np.sqrt(mse)

        if mse < best_mse:
            best_mse = mse
            best_result = {
                "delay_samples": d,
                "Kp": Kp_fit,
                "Kd": Kd_fit,
                "mse": mse,
                "rmse": rmse,
            }

    return best_result


def main():
    parser = argparse.ArgumentParser(
        description="Estimate actuator delay and PD gains from CAN data"
    )
    parser.add_argument("--csv", type=str, default="data.csv", help="Path to CAN bus CSV")
    parser.add_argument("--max-delay", type=int, default=20, help="Max delay in samples to search")
    parser.add_argument("--kd-cap", type=float, default=5.0, help="Cap Kd at this value (motor safety)")
    args = parser.parse_args()

    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)

    # Estimate sample rate
    m1 = df[df.motor_id == 1].sort_values("timestamp")
    dt = m1["timestamp"].diff().median()
    sample_rate = 1.0 / dt

    print(f"Sample rate: {sample_rate:.1f} Hz (dt = {dt * 1000:.2f} ms per sample)")
    print(f"Kd cap: {args.kd_cap}")
    print(f"\nFitting Kp/Kd via least-squares for each delay...")
    print("Model: tau = Kp * (cmd_pos - fb_pos) + Kd * (-fb_vel)")
    print("=" * 100)
    print(f"{'Motor':<15} {'Delay':>6} {'ms':>8} {'Kp':>10} {'Kd':>8} {'RMSE':>10} {'FW Kp':>10} {'FW Kd':>8}")
    print("=" * 100)

    results = {}

    for motor_id in range(1, 13):
        motor_df = df[df["motor_id"] == motor_id].sort_values("timestamp").reset_index(drop=True)

        result = fit_pd_gains_for_motor(
            motor_id,
            motor_df["cmd_pos"].values,
            motor_df["fb_pos"].values,
            motor_df["fb_vel"].values,
            motor_df["fb_cur"].values,
            max_delay=args.max_delay,
            kd_cap=args.kd_cap,
        )

        if result:
            result["motor_id"] = motor_id
            result["delay_ms"] = result["delay_samples"] * dt * 1000
            results[motor_id] = result

            fw_kp = FIRMWARE_KP[motor_id]
            fw_kd = FIRMWARE_KD[motor_id]

            print(f"{JOINT_NAMES[motor_id]:<15} {result['delay_samples']:>6} "
                  f"{result['delay_ms']:>8.1f} {result['Kp']:>10.1f} {result['Kd']:>8.2f} "
                  f"{result['rmse']:>10.2f} {fw_kp:>10.2f} {fw_kd:>8.2f}")

    print("=" * 100)

    # Average by joint type (L/R symmetric)
    print("\n\nIDENTIFIED GAINS (averaged L/R per joint type):")
    print("=" * 80)
    print(f"{'Joint Type':<15} {'Delay':>8} {'Kp':>12} {'Kd':>10} {'RMSE':>10}")
    print("=" * 80)

    joint_types = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
    symmetric_gains = {}

    for jt in joint_types:
        motors = [m for m, t in JOINT_TYPE.items() if t == jt]
        avg_delay = np.mean([results[m]["delay_samples"] for m in motors])
        avg_kp = np.mean([results[m]["Kp"] for m in motors])
        avg_kd = np.mean([results[m]["Kd"] for m in motors])
        avg_rmse = np.mean([results[m]["rmse"] for m in motors])
        symmetric_gains[jt] = {"delay": int(round(avg_delay)), "kp": avg_kp, "kd": avg_kd}
        print(f"{jt:<15} {avg_delay:>8.0f} {avg_kp:>12.1f} {avg_kd:>10.2f} {avg_rmse:>10.2f}")

    print("=" * 80)

    # Output arrays for copy-paste
    print("\n" + "=" * 80)
    print("COPY-PASTE OUTPUT")
    print("=" * 80)

    print("\n# IDENTIFIED_GAINS (from CAN data, Kd capped at {:.1f})".format(args.kd_cap))
    print("IDENTIFIED_GAINS = {")
    for jt in joint_types:
        g = symmetric_gains[jt]
        print(f'    "{jt}": {{"kp": {g["kp"]:.1f}, "kd": {g["kd"]:.1f}, "delay": {g["delay"]}}},')
    print("}")

    # As arrays - order matches policy input/output
    print("\n# As arrays (order matches policy input/output):")
    print("# [L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll,")
    print("#  R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll]")
    kp_list = [round(symmetric_gains[jt]["kp"], 1) for jt in joint_types] * 2
    kd_list = [round(symmetric_gains[jt]["kd"], 1) for jt in joint_types] * 2
    print(f"kp_identified = {kp_list}")
    print(f"kd_identified = {kd_list}")

    # Physics steps
    avg_delay_ms = np.mean([results[m]["delay_ms"] for m in range(1, 13)])
    physics_dt_ms = 2.0  # 500 Hz
    delay_min = max(1, int(np.floor(avg_delay_ms / physics_dt_ms)))
    delay_max = int(np.ceil(avg_delay_ms / physics_dt_ms)) + 1

    print(f"\n# Delay: {avg_delay_ms:.1f} ms = {delay_min}-{delay_max} physics steps at 500Hz")
    print(f"# For DelayedActuatorCfg: delay_min_lag={delay_min}, delay_max_lag={delay_max}")


if __name__ == "__main__":
    main()
