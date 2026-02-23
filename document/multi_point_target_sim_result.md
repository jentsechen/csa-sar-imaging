
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

||binary|CSA (threshold = 0)|threshold = 100|threshold = 200|
|:---:|:---:|:---:|:---:|:---:|
|**image**|![](../diagram/thresholding/island/binary.png)|![](../diagram/thresholding/island/csa.png)|![](../diagram/thresholding/island/threshold_100.png)|![](../diagram/thresholding/island/threshold_200.png)|
|**entropy**||10.6795|10.5306|10.3998|
|**number of non-zero pixels**||1364224|48001|43255|

* Entropy: $E=-\sum_{i=1}^M\sum_{j=1}^N P_{i,j}\ln(P_{i,j}),P_{i,j}=\frac{|I_{i,j}|^2}{\sum_{i=1}^M\sum_{j=1}^N |I_{i,j}|^2}$
    * $\lim_{P\rightarrow0}P\ln(P)=0$

![](../diagram/thresholding/island/entropy.png)
![](../diagram/thresholding/island/n_non_zero_pixel.png)