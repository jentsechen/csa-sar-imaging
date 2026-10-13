| phase | step | calls | mean (s) | total (s) |
|---|---|---|---|---|
| tune | csa | 102 | 0.009 | 0.9 |
| tune | echo+csa | 202 | 0.902 | 182.2 |
| tune | l0 | 1122 | 0.004 | 5.0 |
| tune | l1 | 1122 | 0.004 | 4.8 |
| tune | l12tv | 14576 | 0.543 | 7916.7 |
| tune | yolo_eval_16800 | 1 | 109.579 | 109.6 |
| tune | yolo_eval_9500 | 1 | 61.285 | 61.3 |
| run | csa | 8265 | 0.006 | 49.3 |
| run | echo+csa_2sets | 1393 | 0.895 | 1247.0 |
| run | echo+csa_6sets | 100 | 1.511 | 151.1 |
| run | l0 | 8265 | 0.004 | 33.0 |
| run | l1 | 8265 | 0.004 | 34.3 |
| run | l12tv | 8265 | 0.456 | 3771.5 |
| eval | yolo_eval_100 | 40 | 0.630 | 25.2 |
| eval | yolo_eval_1493 | 20 | 8.786 | 175.7 |

YOLO eval: 6.2 ms per image (incl. per-set overhead)

| full-run part | estimate (h) |
|---|---|
| tune: 100 scenes x (echo + 168 configs) | 2.22 |
| tune: YOLO eval | 0.03 |
| run: 1493 scenes x (echo + 5 SSR x 4 methods) | 1.35 |
| run: variance extra, 100 scenes x 4 seeds x 2 SSR | 0.12 |
| eval: YOLO | 0.06 |
| **total** | **3.77** |
