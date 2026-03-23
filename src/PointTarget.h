#ifndef __point_target__
#define __point_target__

class PointTarget
{
public:
    double range_offset_m, azimuth_offset_sec, scatter_coef;
    PointTarget(double azimuth_offset_sec = 0.0, double range_offset_m = 0.0, double scatter_coef = 0.0);
};

#endif