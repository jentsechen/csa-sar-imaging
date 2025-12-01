#ifndef __point_target__
#define __point_target__

class PointTarget
{
public:
    double range_offset_m, azimuth_offset_sec;
    PointTarget(double azimuth_offset_sec = 0, double range_offset_m = 0);
};

#endif