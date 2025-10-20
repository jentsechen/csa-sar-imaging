import csv
import numpy as np
import json
import subprocess


def read_csv(file_name):
    data = []
    with open(file_name, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data


def gen_input_par():
    input_par = {
        "wavelength_m": 0.4,
        "pulse_width_sec": 10e-6,
        "pulse_rep_freq_hz": 1e3,
        "bandwidth_hz": 50e6,
        "sampling_freq_hz": 64e6,
        "closest_slant_range_m": 4e3,
    }
    with open("input_par.json", "w", encoding="utf-8") as f:
        json.dump(input_par, f)


def get_result():
    # subprocess.run executes the command
    proc = subprocess.run(
        "../build/bin/Debug/TestImagingPar.exe",
        capture_output=True,  # Capture stdout and stderr
        text=True,  # Decode output as text (string)
        check=True,  # Raise an exception for non-zero exit codes (errors)
    )
    print("\n--- C++ Program Output (STDOUT) ---")
    print(proc.stdout)
    with open("output_par.json", "r", encoding="utf-8") as f:
        result = json.load(f)
    rng_t = np.array(result["range_time_axis_sec"])
    rng_f = np.array(result["range_freq_axis_hz"])
    azm_t = np.array(result["azimuth_time_axis_sec"])
    azm_f = np.array(result["azimuth_freq_axis_hz"])
    return rng_t, rng_f, azm_t, azm_f


def col2row(col_arr):
    row_arr = []
    for c in col_arr:
        row_arr.append(c[0])
    return row_arr


def str2float(str_arr):
    float_arr = []
    for s in str_arr:
        float_arr.append(float(s))
    return np.array(float_arr)


def get_golden():
    rng_t = str2float(read_csv("range_time_axis_sec.csv")[0])
    rng_f = str2float(read_csv("range_freq_axis_hz.csv")[0])
    azm_t = str2float(col2row(read_csv("azimuth_time_axis_sec.csv")))
    azm_f = str2float(read_csv("azimuth_freq_axis_hz.csv")[0])
    return rng_t, rng_f, azm_t, azm_f


def check(result, golden):
    print("error of range_time_axis_sec: {}".format(sum(abs(result[0] - golden[0]))))
    print("error of range_freq_axis_hz: {}".format(sum(abs(result[1] - golden[1]))))
    print("error of azimuth_time_axis_sec: {}".format(sum(abs(result[2] - golden[2]))))
    print("error of azimuth_freq_axis_hz: {}".format(sum(abs(result[3] - golden[3]))))


if __name__ == "__main__":
    gen_input_par()
    rng_t_r, rng_f_r, azm_t_r, azm_f_r = get_result()
    rng_t_g, rng_f_g, azm_t_g, azm_f_g = get_golden()
    check((rng_t_r, rng_f_r, azm_t_r, azm_f_r), (rng_t_g, rng_f_g, azm_t_g, azm_f_g))
