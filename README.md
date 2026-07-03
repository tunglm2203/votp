<div><h2>[ICML'26 Oral] Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning</h2></div>
<br>

**Tung M. Luu, Hwanhee Kim, Younghwan Lee, Chang D. Yoo**
<br>
KAIST, South Korea
<br>
[[Paper]](https://arxiv.org/abs/2606.16856v1) [[Website]](https://votp2026.github.io/) 


## Overview
This is the official implementation of **Video-based Optimal Transport Preference (VOTP)**.

## Installation

```
conda create --name votp python=3.9
conda activate votp
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e external_packages/Metaworld --no-deps
pip install -e external_packages/d4rl --no-deps
```

## Download Dataset:
```
bash bash_scripts/download_data.sh
```

## Run Experiments
```
bash bash_scripts/run_mw_votp.sh
```

## Citation
If you use this repo in your research, please consider citing the paper as follows:
```
@inproceedings{
    luu2026video,
    title={Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning},
    author={Tung M. Luu and Hwanhee Kim and Younghwan Lee and Chang D. Yoo},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=G8LVO5easu}
}
```

## Acknowledgements
- This work was supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2021II211381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-24742969, Intelligent Robotic System using Continual Learning and Multimodal Language Model based Multi Attribute Feedback).

- This repo contains code adapted from [IPL](https://github.com/jhejna/inverse-preference-learning/) and 
[CPL](https://github.com/jhejna/cpl). We thank the authors and contributors for open-sourcing their code.

## License

MIT