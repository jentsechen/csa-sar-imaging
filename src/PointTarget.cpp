#include "PointTarget.h"

PointTarget::PointTarget(double azimuth_offset_sec, double range_offset_m, double scatter_coef)
    : azimuth_offset_sec(azimuth_offset_sec), range_offset_m(range_offset_m), scatter_coef(scatter_coef) {}