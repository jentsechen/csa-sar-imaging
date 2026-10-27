# Clean-Scene Amplitude Distribution: Sea Clutter vs Ships

`python3 scene_statistics.py` over all 1593 offshore scenes (`images/`, HRSID test
split), 1.02e9 pixels, about 1 minute. Scene amplitude = grayscale pixel value, the
same value the echo simulation uses as scatterer amplitude (HRSID pixels are linear
amplitude).

Pixel classes come from the HRSID instance-segmentation polygons
(`sar_ship_detect/HRSID_JPG/annotations/test2017.json`), not the GT boxes: a box is
on average only 0.50 ship (median 0.48, 10th percentile 0.33).

- ship = inside a polygon
- sea = farther than 10 px from every polygon (guard band against ship edges,
  sidelobes and ambiguity streaks)

Figure: `figures/amplitude_hist.png`. Each curve is the percentage of its own class's
pixels at each amplitude (sums to 100 %, log y axis from 0.0001 % to 100 %), so the curves compare shapes; at any amplitude the sea still has far
more pixels than the ships (about 500x overall). Numbers: `stats.json`.

## Results

| | ship | sea (guard 10 px) |
|---|---|---|
| fraction of pixels | 0.20 % | 99.35 % |
| mean | 132.3 | 15.0 |
| median | 115 | 13 |
| 90th / 99th / 99.9th percentile | 255 / 255 / 255 | 28 / 47 / 69 |
| clipped at 255 | 12.0 % | 0.0 % |

Sea Rayleigh fit (maximum likelihood): σ = 13.4. Fraction of sea pixels above t:

| t | measured | Rayleigh |
|---|---|---|
| 40 | 2.3e-2 | 9.1e-3 |
| 47 | 9.9e-3 | 1.6e-3 |
| 60 | 2.3e-3 | 3.0e-5 |
| 80 | 3.9e-4 | 1.1e-8 |

## Findings

1. **Sea clutter is a Rayleigh-like hump, not zeros.** The sea peaks at amplitude 9
   (median 13) and follows a Rayleigh law up to about 40; ships spread from about 20 to
   255 (median 115). Refinement that zeroes low amplitudes removes this hump, which is
   the background against which the detector has to find ships.
2. **The sea tail is much heavier than Rayleigh.** Above about 50 the measured sea
   fraction exceeds the Rayleigh prediction by 6x at t = 47 and by 75x at t = 60. A
   threshold set from a Rayleigh clutter model (CFAR-style) would leave far more bright
   sea pixels than predicted.
3. **8-bit clipping.** 12 % of ship pixels sit at 255, so the true dynamic range of
   ships is larger than the images show.

Caveat: in the simulation the sea clutter is part of the "clean" scene, so a
refinement that zeroes the sea removes real clutter as well as added noise.
