# Building Extraction from Remote Sensing Imagery under Adverse Weather: Benchmark and Baseline

This is the official implementation of the paper:<br/>
> **Building Extraction from Remote Sensing Imagery under Adverse Weather: Benchmark and Baseline**
>
> *Feifei Sang†, Wei Lu†, Hongruixuan Chen, Si-Bao Chen∗, Jin Tanga and Bin Luoa* 

----

<p align="center"> 
<img src="./fig.png" width=100% 
class="center">
<p align="center">  Illustration of the annotation pipeline for the HaLoBuilding dataset based on the Same-Scene Multitemporal Pairing strategy. The process consists of two stages: Stage 1 pairs clear reference images with adverse-weather counterparts. Stage 2 transfers high-quality labels from clear images, followed by rigorous manual refinement. The yellow dashed boxes highlight inconsistent regions caused by temporal gaps or visibility degradation, which are manually corrected to ensure high-fidelity, pixel-level alignment with the actually visible buildings.
</p> 

----

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Building extraction from Remote Sensing (RS) imagery is crucial for smart city construction and disaster emergency response. However, existing optical methods and benchmarks focus primarily on ideal clear-weather conditions. Consequently, they exhibit substantial performance degradation in real-world adverse scenarios, where atmospheric scattering and illumination degradation are prevalent. While SAR offers all-weather sensing, its side-looking geometry causes geometric distortions; conversely, optical imagery retains superior geometric fidelity and possesses latent semantic information. However, a dedicated optical benchmark tailored for building extraction under adverse weather is still lacking. To address these challenges, we introduce HaLoBuilding, the first optical benchmark specifically designed for building extraction under hazy and low-light conditions. By leveraging a same-scene multitemporal pairing strategy, we ensure pixel-level label alignment and high fidelity even under extreme degradation. Building upon this benchmark, we propose HaLoBuild-Net, a novel end-to-end framework for building extraction in adverse RS scenarios. At its core, we develop a Spatial-Frequency Filtering Module (SFFM) to effectively disentangle meteorological interference from building structures by coupling large receptive field attention with frequency-domain low-frequency perception. Additionally, a Global Multi-scale Guidance Module (GMGM) provides global semantic constraints to anchor building topologies, while a Mutual-Guided Fusion Module (MGFM) bridges the semantic gap between the encoder and decoder through bidirectional interaction. Extensive experiments demonstrate that HaLoBuild-Net significantly outperforms state-of-the-art methods and conventional cascaded restoration-segmentation paradigms on the HaLoBuilding dataset, while maintaining robust generalization on WHU, INRIA, and LoveDA datasets.
</details>


## Installation
+ Prerequisites for Python:
  - Creating a virtual environment in the terminal: `conda create -n HaLoBuild python=3.10`
  - Activate the environment: `conda activate HaLoBuild`
  - Installing necessary packages: `pip install -r requirements.txt`

+ Train/Test
  - `python -m tools.train`
  - `python -m tools.test`


## Introduction

The code will be available.

The dataset can be downloaded at [Baidu netdisk](https://pan.baidu.com/s/1MDS15v0zWMHhTf9mAnZxrw?pwd=0220)(Password:0220) or [Google Drive](https://drive.google.com/file/d/1fT5fI-OQGl62Qz2eADtftifs6je1YR82/view?usp=sharing).

If you have any questions about this work, you can contact me. 

Email: [luwei_ahu@qq.com](mailto:luwei_ahu@qq.com); WeChat: luwei_ahu.

Your star is the power that keeps us updating github.

## Citation
If HaLoBuild-Net or the HaLoBuilding dataset is useful or relevant to your research, please kindly recognize our contributions by citing our paper::
```
@article{wang2026multi,
  title={Multi-Modal Building Change Detection for Large-Scale Small Changes: Benchmark and Baseline},
  author={Wang, Ye and Lu, Wei and You, Zhihui and Chen, Keyan and Liu, Tongfei and Li, Kaiyu and Chen, Hongruixuan and Shu, Qingling and Chen, Sibao},
  journal={arXiv preprint arXiv:2603.19077},
  year={2026}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
