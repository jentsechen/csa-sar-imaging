import os
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

from generate_point_target import read_sar_slice, segment


class DenoisingMethod(Enum):
    RAW       = "raw"
    BINARY    = "binary"
    OTSU      = "otsu"
    BILATERAL = "bilateral"
    GUIDED    = "guided"
    TV        = "tv"
    LEE       = "lee"

    def segment_kwargs(self, threshold=None):
        """Return (method_str, kwargs) to pass to segment(), None method_str means raw."""
        if self is DenoisingMethod.RAW:
            return None, {}
        if self is DenoisingMethod.BINARY:
            return "binary", {"threshold": threshold}
        if self is DenoisingMethod.OTSU:
            return "otsu", {"morph_close": True}
        return self.value, {"morph_close": True, "preserve_gray": True}

    def display_label(self, threshold=None):
        labels = {
            DenoisingMethod.RAW:       "Raw SAR Image",
            DenoisingMethod.BINARY:    f"Binary Thresholding (threshold = {threshold})",
            DenoisingMethod.OTSU:      "Otsu's Thresholding + Morphological Closing",
            DenoisingMethod.BILATERAL: "Bilateral Filter + Morphological Closing",
            DenoisingMethod.GUIDED:    "Guided Filter + Morphological Closing",
            DenoisingMethod.TV:        "Total Variation (TV) Filter + Morphological Closing",
            DenoisingMethod.LEE:       "Lee Filter + Morphological Closing",
        }
        return labels[self]


def compare_all_methods(scene_name, file_name, row_start, row_len, col_start, col_len,
                        methods=None, threshold=110,
                        output_dir="../diagram/umbra_image_denoising"):
    """
    Run selected denoising methods on a SAR slice, save per-method images,
    and export a LaTeX table of non-zero sample counts.

    Args:
        scene_name:  prefix used for output file names
        file_name:   input GeoTIFF path
        row_start, row_len, col_start, col_len: slice bounds
        methods:     list of DenoisingMethod members to run.
                     None (default) runs all methods.
        threshold:   fixed threshold used for DenoisingMethod.BINARY
        output_dir:  directory where PNGs are written (created if absent)
    """
    if methods is None:
        methods = list(DenoisingMethod)

    os.makedirs(output_dir, exist_ok=True)
    sar_slice = read_sar_slice(file_name, row_start, row_len, col_start, col_len)

    results = []
    for m in methods:
        method_str, kwargs = m.segment_kwargs(threshold)
        data = sar_slice if method_str is None else segment(sar_slice, method=method_str, **kwargs)
        arr = np.array(data)
        results.append((m, arr, int((arr != 0).sum())))

    # individual images
    for m, arr, _ in results:
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(np.flipud(arr), origin='lower', cmap='viridis')
        ax.set_xlabel('range')
        ax.set_ylabel('azimuth')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{scene_name}_{m.value}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # non-zero count summary
    print(f"\n{'Method':<55} {'Non-Zeros':>10}")
    print("-" * 67)
    for m, _, num_nonzero in results:
        print(f"{m.display_label(threshold):<55} {num_nonzero:>10,}")

    # export LaTeX table
    tex_rows = "\n".join(
        f"        {m.display_label(threshold)} & {num_nonzero:,} \\\\"
        for m, _, num_nonzero in results
    )
    tex_table = (
        "\\begin{table}[H]\n"
        "    \\centering\n"
        "    \\caption{Non-zero sample counts per denoising method.}\n"
        "    \\begin{tabular}{lr}\n"
        "        \\toprule\n"
        "        Method & Non-zero samples \\\\\n"
        "        \\midrule\n"
        f"{tex_rows}\n"
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        "\\end{table}\n"
    )
    tex_path = os.path.join(os.path.dirname(output_dir),
                            "../document/simulation_method/umbra_denoising_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex_table)
    print(f"Images saved to:      {output_dir}")
    print(f"LaTeX table saved to: {tex_path}")
