<div><h2>[ICML'26 Oral] Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning</h2></div>
<br>

**[Tung M. Luu](https://tungluu94.github.io/), [Hwanhee Kim](https://khhandrea.github.io/), [Younghwan Lee](https://scholar.google.com/citations?user=e61HtZUAAAAJ&hl=en), [Chang D. Yoo](https://sanctusfactory.com/family.php)**
<br>
KAIST, South Korea
<br>

<p align="center">
        <a href="https://votp2026.github.io/" target='_blank'>
        <img src="https://img.shields.io/badge/🌐-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/pdf/2606.16856v1" target='_blank'>
        <img src="https://img.shields.io/badge/2026-ICML-brightgreen">
        </a>
        <a href="https://arxiv.org/abs/2606.16856v1" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2606.16856-b31b1b.svg">
        </a>
</p>

<p align="center">
    <img src="https://github.com/tunglm2203/votp/blob/main/assets/votp-overview.png" alt="Overview of the VOTP method" width="700">
</p>

## Overview
This is the official implementation of **Video-based Optimal Transport Preference (VOTP)**.

**TL;DR**: VOTP is a feedback-efficient offline PbRL method. It treats trajectory segments as distributions of video foundation model features and uses optimal transport to propagate a handful of preference labels across the whole dataset, yielding pseudo-labels that train a Bradley–Terry reward for offline RL. With only ~10 labels it matches or exceeds methods using far more feedback.


## ⚙️ Environmental Setups

```sh
git clone https://github.com/tunglm2203/votp.git
cd votp

# install votp environment
conda create --name votp python=3.9
conda activate votp
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e external_packages/Metaworld --no-deps
pip install -e external_packages/d4rl --no-deps
```

## 📁 Data Preparations
For MetaWorld, we build on the preference dataset from [CPL](https://github.com/jhejna/cpl), and for D4RL Gym, we build on the preference dataset from [Preference Transformer](https://github.com/csmile-1006/PreferenceTransformer). To download preprocessed data, please run the following command:
```
bash bash_scripts/download_data.sh
```

## 🚀 Get Started
```
# check if environment is activated properly
conda activate votp

# run 4 MetaWorld tasks with 5 seeds each. Details can be found in script.
bash bash_scripts/run_mw_votp.sh
```

## ⭐ Citation
If you find our repository useful, please consider giving it a star ⭐ and citing our research papers in your work:
```
@inproceedings{
    luu2026video,
    title={Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning},
    author={Tung M. Luu and Hwanhee Kim and Younghwan Lee and Chang D. Yoo},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026}
}
```

## Acknowledgements
- This work was supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2021-II211381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-24742969, Intelligent Robotic System using Continual Learning and Multimodal Language Model based Multi Attribute Feedback).

- This repo contains code adapted from [IPL](https://github.com/jhejna/inverse-preference-learning/) and 
[CPL](https://github.com/jhejna/cpl). We thank the authors and contributors for open-sourcing their code.

## License

MIT