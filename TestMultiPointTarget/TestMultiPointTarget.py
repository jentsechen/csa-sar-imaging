import csv
import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objects as go
from enum import Enum, auto

class Scene(Enum):
    Basic, Coast, Island = auto(), auto(), auto()

def gen_input_par(scene):
    if scene == Scene.Basic:
        input_par = {
            "wavelength_m": 0.4,
            "pulse_width_sec": 10e-6,
            "pulse_rep_freq_hz": 1e3,
            "bandwidth_hz": 50e6,
            "sampling_freq_hz": 64e6,
            "closest_slant_range_m": 4e3,
            "azi_win_en": True,
            "rng_pad_time": 8,
            "noise_en": False,
            "snr_db": 25.0
        }
    elif scene == Scene.Coast:
        input_par = {
            "wavelength_m": 0.08008,
            "pulse_width_sec": 4.34375e-6,
            "pulse_rep_freq_hz": 1e3,
            "bandwidth_hz": 50e6,
            "sampling_freq_hz": 64e6,
            "closest_slant_range_m": 4e3,
            "azi_win_en": True,
            "rng_pad_time": 8,
            "noise_en": False,
            "snr_db": 25.0
        }
    elif scene == Scene.Island:
        input_par = {
            "wavelength_m": 0.042048,
            "pulse_width_sec": 2.28125e-6,
            "pulse_rep_freq_hz": 1e3,
            "bandwidth_hz": 50e6,
            "sampling_freq_hz": 64e6,
            "closest_slant_range_m": 4e3,
            "azi_win_en": True,
            "rng_pad_time": 8,
            "noise_en": False,
            "snr_db": 25.0
        }
    else:
        print("The scene is not supported!")
    with open("input_par.json", "w", encoding="utf-8") as f:
        json.dump(input_par, f)


def run_cpp(args=[]):
    cmd = ["../build/TestMultiPointTarget", args[0], args[1]]
    proc = subprocess.run(
        cmd,
        capture_output=True,  # Capture stdout and stderr
        text=True,  # Decode output as text (string)
        check=True,  # Raise an exception for non-zero exit codes (errors)
    )
    print("\n--- C++ Program Output (STDOUT) ---")
    print(proc.stdout)


def save_3d_plot(source_folder_name, source_file_name):
    single_point_target = np.load(
        "./{}/{}.npy".format(source_folder_name, source_file_name)
    )
    fig = go.Figure(
        data=[
            go.Surface(
                z=single_point_target[
                    (5555 - 400) : (5555 + 400), (2560 - 160) : (2560 + 160)
                ],
                colorscale="Viridis",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="range direction",
            yaxis_title="azimuth direction",
            zaxis_title="magnitude (dB)",
        )
    )
    fig.write_html("./{}/{}.html".format(source_folder_name, source_file_name))


def save_3d_plot_of_focused_image(source_folder_name, source_file_name):
    single_point_target = np.load(
        "./{}/{}.npy".format(source_folder_name, source_file_name)
    )
    fig = go.Figure(
        data=[
            go.Surface(
                z=single_point_target,
                colorscale="Viridis",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="range direction",
            yaxis_title="azimuth direction",
            zaxis_title="magnitude (dB)",
        )
    )
    fig.write_html("./{}/{}.html".format(source_folder_name, source_file_name))


if __name__ == "__main__":
    # run_cpp(args=["iter_recov", "test"])
    # run_cpp(args=["calc_mag", "./iter_result/csa_out_iter_0"])
    # run_cpp(args=["calc_mag", "./iter_result/csa_out_iter_1"])
    # run_cpp(args=["calc_mag", "./iter_result_down_smp/csa_out_iter_0"])
    # run_cpp(args=["calc_mag", "./iter_result_down_smp/csa_out_iter_1"])
    # for i in range(5):
    #     run_cpp(
    #         args=[
    #             "calc_mag",
    #             "./iter_result_multi_point_rng_dpl_anal/csa_out_iter_{}".format(i),
    #         ]
    #     )
    # save_3d_plot("iter_result_down_smp", "csa_out_iter_0_mag_db")
    # save_3d_plot("iter_result_down_smp", "csa_out_iter_1_mag_db")
    # for i in range(5):
    #     save_3d_plot(
    #         "iter_result_multi_point_rng_dpl_anal", "csa_out_iter_{}_mag_db".format(i)
    #     )

    # single_point_target = np.load("./focused_image/single_point_target_mag_db.npy")
    # fig = go.Figure(
    #     data=[
    #         go.Surface(
    #             z=single_point_target[
    #                 (5555 - 400) : (5555 + 400), (2560 - 160) : (2560 + 160)
    #             ],
    #             colorscale="Viridis",
    #         )
    #     ]
    # )
    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title="range direction",
    #         yaxis_title="azimuth direction",
    #         zaxis_title="magnitude (dB)",
    #     )
    # )
    # fig.write_html("single_point_target.html")

    # multi_point_target = np.load("./focused_image/multi_point_target_mag_db.npy")
    # fig = go.Figure(
    #     data=[
    #         go.Surface(
    #             z=multi_point_target[
    #                 (5555 - 400) : (5555 + 400), (2560 - 160) : (2560 + 160)
    #             ],
    #             colorscale="Viridis",
    #         )
    #     ]
    # )
    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title="range direction",
    #         yaxis_title="azimuth direction",
    #         zaxis_title="magnitude (dB)",
    #     )
    # )
    # fig.write_html("multi_point_target.html")

    # gen_input_par(scene=Scene.Coast)
    gen_input_par(scene=Scene.Island)
    run_cpp(args=["gen", "multi_point_target"])
    run_cpp(args=["focus", "multi_point_target_image"])
    run_cpp(args=["calc_mag", "./focused_image/multi_point_target_image"])
    # save_3d_plot_of_focused_image("focused_image", "multi_point_target_image_mag_db")
    # save_3d_plot("focused_image", "multi_point_target_image_mag_db")

    # run_cpp(args=["iter_recov", "test"])
    # for i in range(5):
    #     run_cpp(
    #         args=[
    #             "calc_mag",
    #             "./iter_result_multi_point_image/csa_out_iter_{}".format(i),
    #         ]
    #     )
    # for i in range(5):
    #     save_3d_plot_of_focused_image(
    #         "./iter_result_multi_point_image", "csa_out_iter_{}_mag_db".format(i)
    #     )

    # gen_input_par(scene=Scene.Basic)
    # run_cpp(args=["gen", "single_point_target"])
    # run_cpp(args=["focus", "single_point_target"])
    # run_cpp(args=["calc_mag", "./focused_image/single_point_target"])
    # save_3d_plot("focused_image", "single_point_target_mag_db")