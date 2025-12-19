import csv
import numpy as np
import json
import subprocess
import scipy.io

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


def run_cpp(test_echo_sig_en=False):
    cmd = ["../build/TestImagingPar"]
    if test_echo_sig_en:
        cmd.append("test_echo_sig")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    print("\n--- C++ Program Output (STDOUT) ---")
    print(proc.stdout)


def get_constructor_result():
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


def get_constructor_golden():
    rng_t = str2float(read_csv("range_time_axis_sec.csv")[0])
    rng_f = str2float(read_csv("range_freq_axis_hz.csv")[0])
    azm_t = str2float(col2row(read_csv("azimuth_time_axis_sec.csv")))
    azm_f = str2float(read_csv("azimuth_freq_axis_hz.csv")[0])
    return rng_t, rng_f, azm_t, azm_f


def check_constructor():
    result = get_constructor_result()
    golden = get_constructor_golden()
    print("error of range_time_axis_sec: {}".format(sum(abs(result[0] - golden[0]))))
    print("error of range_freq_axis_hz: {}".format(sum(abs(result[1] - golden[1]))))
    print("error of azimuth_time_axis_sec: {}".format(sum(abs(result[2] - golden[2]))))
    print("error of azimuth_freq_axis_hz: {}".format(sum(abs(result[3] - golden[3]))))


def check_gen_point_target_echo_signal():
    golden = scipy.io.loadmat("echo_signal_golden.mat")["point_target_echo_signal"]
    result = np.load("./echo_signal_result.npy")
    print(
        "error of point_target_echo_signal: {}".format(sum(sum(abs(result - golden))))
    )


if __name__ == "__main__":
    gen_input_par()
    run_cpp(test_echo_sig_en=True)
    check_constructor()
    check_gen_point_target_echo_signal()
