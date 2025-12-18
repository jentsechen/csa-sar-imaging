import numpy as np
import json
import subprocess
import scipy.io
import plotly.graph_objs as go
import plotly.offline as pof
from plotly.subplots import make_subplots

if __name__ == "__main__":
    # folder_name = "iter_result_multi_point"
    # folder_name = "iter_result_multi_point_rng_dpl_anal"
    # folder_name = "iter_result_single_point"
    # iter_0 = np.load("./{}/csa_out_iter_0.npy".format(folder_name))
    # iter_1 = np.load("./{}/csa_out_iter_1.npy".format(folder_name))
    # iter_2 = np.load("./{}/csa_out_iter_2.npy".format(folder_name))
    # iter_3 = np.load("./{}/csa_out_iter_3.npy".format(folder_name))
    # iter_4 = np.load("./{}/csa_out_iter_4.npy".format(folder_name))

    # iter_0 = np.load("./iter_result_multi_point/csa_out_iter_0.npy")
    # iter_1 = np.load("./iter_result_multi_point/csa_out_iter_1_mag_db.npy")
    # iter_2 = np.load("./iter_result_multi_point/csa_out_iter_2_mag_db.npy")
    # iter_3 = np.load("./iter_result_multi_point/csa_out_iter_3_mag_db.npy")
    # iter_4 = np.load("./iter_result_multi_point/csa_out_iter_4_mag_db.npy")
    # iter_5 = np.load("./iter_result_multi_point/csa_out_iter_5_mag_db.npy")

    # print(sum(sum(iter_0 != iter_1)))
    # print(sum(sum(iter_1 != iter_2)))
    # print(sum(sum(iter_2 != iter_3)))
    # print(sum(sum(iter_3 != iter_4)))

    # diff_mat = iter_0 != iter_1
    # # print(np.where(diff_mat.any(axis=0))[0])
    # # print(np.where(diff_mat.any(axis=1))[0])
    # if 2560 in np.where(diff_mat.any(axis=0))[0]:
    #     print("TRUE")

    # print(sum(iter_0[:, 2560] != iter_1[:, 2560]))
    # print(sum(iter_1[:, 2560] != iter_2[:, 2560]))
    # print(sum(iter_2[:, 2560] != iter_3[:, 2560]))
    # print(sum(iter_3[:, 2560] != iter_4[:, 2560]))
    # print(sum(iter_4[:, 2560] != iter_5[:, 2560]))

    # print(sum(sum(abs(iter_0 - iter_1))))
    # print(sum(sum(abs(iter_1 - iter_2))))
    # print(sum(sum(abs(iter_2 - iter_3))))
    # print(sum(sum(abs(iter_3 - iter_4))))

    # figure = make_subplots(rows=1, cols=1)
    # figure.add_trace(go.Scatter(y=iter_0[:, 2560]), row=1, col=1)
    # figure.add_trace(go.Scatter(y=iter_1[:, 2560]), row=1, col=1)
    # figure.add_trace(go.Scatter(y=iter_2[:, 2560]), row=1, col=1)
    # figure.add_trace(go.Scatter(y=iter_3[:, 2560]), row=1, col=1)
    # figure.add_trace(go.Scatter(y=iter_4[:, 2560]), row=1, col=1)
    # figure.add_trace(go.Scatter(y=iter_5[:, 2560]), row=1, col=1)
    # # figure.update_layout(
    # #     xaxis=dict(title="sample"),
    # #     yaxis=dict(title="amplitude (real part)"),
    # #     xaxis2=dict(title="sample"),
    # #     yaxis2=dict(title="amplitude (imaginary part)"),
    # #     font=dict(size=20),
    # # )
    # figure.write_html("image_diff_in_db.html")

    # figure = make_subplots(rows=1, cols=1)
    # figure.add_trace(go.Scatter(y=iter_1[:, 2560] - iter_0[:, 2560]), row=1, col=1)
    # figure.add_trace(go.Scatter(y=iter_2[:, 2560] - iter_1[:, 2560]), row=1, col=1)
    # figure.add_trace(go.Scatter(y=iter_3[:, 2560] - iter_2[:, 2560]), row=1, col=1)
    # figure.add_trace(go.Scatter(y=iter_4[:, 2560] - iter_3[:, 2560]), row=1, col=1)
    # figure.add_trace(go.Scatter(y=iter_5[:, 2560] - iter_4[:, 2560]), row=1, col=1)
    # figure.write_html("image_diff_in_db.html")

    golden = np.load("./focused_image/multi_point_target_image.npy")
    folder_name = "iter_result_multi_point_image"
    iter_result = []
    for iter in range(10):
        iter_result.append(np.load("./{}/csa_out_iter_{}.npy".format(folder_name, iter)))

    print("not equal:")
    for iter in range(9):
        print(sum(sum(iter_result[iter] != iter_result[iter + 1])))
    print("")

    print("difference:")
    for iter in range(9):
        print(sum(sum(abs(iter_result[iter] - iter_result[iter + 1]))))

    print("difference:")
    for iter in range(10):
        print(sum(sum(abs(iter_result[iter] - golden))))

    print("DONE")
