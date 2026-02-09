
## Azimuth Compression With Variant Range
* azimuth compression filter with fixed range
    * location $T_3$: azimuth time offset is 0.04 sec, range distance offset is 30 m

|point target at scene center $T_0$|point target at location $T_3$|
|:---:|:---:|
|![](../diagram/fixed_range_scene_center/image.png)|![](../diagram/fixed_range_diff_azi_rng/image.png)|
|![](../diagram/fixed_range_scene_center/rng_slice.png)|![](../diagram/fixed_range_diff_azi_rng/rng_slice.png)|
|![](../diagram/fixed_range_scene_center/azi_slice.png)|![](../diagram/fixed_range_diff_azi_rng/azi_slice.png)|

* azimuth compression filter with variant range
    * location $T_3$: azimuth time offset is 0.04 sec, range distance offset is 30 m

|point target at scene center $T_0$|point target at $T_3$|
|:---:|:---:|
|![](../diagram/var_range_scene_center/image.png)|![](../diagram/var_range_diff_azi_rng/image.png)|
|![](../diagram/var_range_scene_center/rng_slice.png)|![](../diagram/var_range_diff_azi_rng/rng_slice.png)|
|![](../diagram/var_range_scene_center/azi_slice.png)|![](../diagram/var_range_diff_azi_rng/azi_slice.png)|

## Scaling
|full scale|30 dB dynamic range|ground truth|
|:---:|:---:|:---:|
|![](../diagram/scaling/coast_full_scale.png)|![](../diagram/scaling/coast_30_db_dynamic_range.png)|![](../diagram/scaling/point_target_location.png)|

## Effect of Thresholding Operator
![](../diagram/thresholding/island/original.png)

|binary|CSA|thresholding|
|:---:|:---:|:---:|
|![](../diagram/thresholding/island/binary.png)|![](../diagram/thresholding/island/csa.png)|![](../diagram/thresholding/island/thresholding.png)|