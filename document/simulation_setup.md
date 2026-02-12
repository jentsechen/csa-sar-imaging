## Signal Model
* Multi-target scattering model of chirp signal: $S=[s_{m,n}]_{M\times N}$
    * $\eta=m\Delta\eta$: Slow time
    * $\tau=m\Delta\tau$: Fast time
    * $A_k$: Scattering coefficient of point target $k$
    * $R_k(m\Delta\eta)=\sqrt{(R_0+\Delta R_{0,k})^2+(m\Delta\eta+\Delta\eta_k)^2V_r^2}$
        * $\Delta R_{0,k}$: Slant range offset from point target $k$ to scene center
        * $\Delta\eta_k$: Azimuth time offset from point target $k$ to scene center

$$s_{m,n}=\sum_k A_{k}
\text{rect}\left(\frac{n\Delta\tau-\frac{2R_k(m\Delta\eta)}{2}}{T_r}\right)
\cdot e^{j\pi\cdot\frac{B_r}{T_r}\cdot\left(n\Delta\tau-\frac{2R_k(m\Delta\eta)}{2}\right)^2}
\cdot e^{-j4\pi\frac{R_k(m\Delta\eta)}{\lambda}}$$

* Coherent scattering in time domain: $S\circ C+W$
    * Time domain: $C=[c_{m,n}]_{M\times N}$ , $c_{m,n}=e^{j\theta_{m,n}}, \theta_{m,n} \sim \mathcal{U}(0,2\pi)$
* Coherent scattering in spatial domain: $\mathcal{T}(S+W)\circ C$
    * $C=[c_{m,n}]_{M\times N}$ , $c_{m,n} \sim \mathcal{E}(1)$
    * $\mathcal{T}(\cdot)$: Imaging algorithm, e.g. RDA or CSA
* Thermal noise (time domain): $W=[w_{m,n}]_{M\times N}$ , $w_{m,n} \sim \mathcal{CN}(0, \sigma)$
    * $\sigma^2=P_n/2, P_n=P_s/SNR$

## How to Convert Image to Echo Signal
![](../diagram/drawio/image_to_echo.svg)
![](../diagram/drawio/multi_point_target.svg)


