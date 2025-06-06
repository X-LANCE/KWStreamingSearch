# KWStreamingSearch

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2403.13332-b31b1b.svg)](https://arxiv.org/abs/2403.13332)
[![arXiv](https://img.shields.io/badge/arXiv-2412.12635-b31b1b.svg)](https://arxiv.org/abs/2412.12635)


## 📚 Publications
Official repository for streaming decoding algorithms for ASR-based Keyword Spotting (KWS) systems, featuring implementations from our research papers:

1. **MFA-KWS**: Effective Keyword Spotting with Multi-head Frame-asynchronous Decoding 
   (**arxiv**, [Arxiv](https://arxiv.org/abs/2505.19577) | (**T-ASLP Under Review**))
2. **CDC-KWS**: Streaming Keyword Spotting Boosted by Cross-layer Discrimination Consistency 
   (**ICASSP 2025**, [Arxiv](https://arxiv.org/abs/2412.12635) | [IEEE](https://ieeexplore.ieee.org/document/10890010))
1. **TDT-KWS**: Fast and Accurate Keyword Spotting Using Token-and-Duration Transducer
   (**ICASSP 2024**, [Arxiv](https://arxiv.org/abs/2403.13332) | [IEEE](https://ieeexplore.ieee.org/document/10446909))

## 📖 Overview

This repository contains state-of-the-art streaming decoding algorithms for KWS systems, featuring:

- **Frame-asynchronous decoding** for efficient keyword search
- **Multi-head fusion** for robust performance
- **Cross-layer consistency** for improved discrimination
- **Transducer-based** approaches for Transducer-based KWS
- **CTC-based** approaches for CTC-based KWS 

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/KWStreamingSearch.git
cd KWStreamingSearch
pip install -r requirements.txt
```

## 🏗️ Repository Structure

```
KWStreamingSearch
├── KWStreamingSearch
│   ├── CTC
│   │   ├── cdc_streaming_search.py  # CDC-enhanced streaming search
│   │   └── ctc_streaming_search.py  # Basic CTC streaming search
│   ├── MFA
│   │   ├── mfa_streaming_search.py  # Multi-head frame-asynchronous search
│   │   └── mfs_streaming_search.py  # Multi-head frame-synchronous search
│   ├── Transducer
│   │   └── trans_streaming_search.py # Transducer-based streaming search
│   ├── __init__.py
│   ├── base_search.py  # Base search class
│   ├── example.py  # Usage examples
│   └── fusion_strategy.py  # Fusion strategies
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

See `example.py` for more detailed usage examples.

## 📊 Performance Highlights

- **TDT-KWS**:
  - Propose a new Transducer-based streaming deocding method, outperforming the traditional ASR-based decoding.
  - Significant inference speed-up by the variant of Token-and-Duration Transducer.

- **CDC-KWS**:
  - CTC-based keyword-specific streaming search.
  - Improved robustness in noisy environments by cross-layer discrimination consitency.

- **MFA-KWS**:
  - State-of-the-art on Snips, MobvoiHotwords, LibriKWS-20.
  - Significant speed-up over frame-synchronous baselines.

## 🤝 Contributing

We welcome contributions! Please open an issue or submit a pull request.

## 📜 License

This project is licensed under the Apache License 2.0

## 📝 Citation

If you think our work helps in your research, please cite:

```bibtex
@inproceedings{icassp2024-yuxi-tdt_kw,
  author       = {Yu Xi and Hao Li and Baochen Yang and Haoyu Li and Hainan Xu and Kai Yu},
  title        = {{TDT-KWS:} Fast and Accurate Keyword Spotting Using Token-and-Duration
                  Transducer},
  booktitle    = {ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages        = {11350--11355},
  year         = {2024},
}

@INPROCEEDINGS{icassp2025-yuxi-cdc_kws,
  author={Xi, Yu and Li, Haoyu and Gu, Xiaoyu and Li, Hao and Jiang, Yidi and Yu, Kai},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Streaming Keyword Spotting Boosted by Cross-layer Discrimination Consistency}, 
  year={2025},
  pages={1-5},
}
```

(MFA-KWS will be added after publication)

## 📧 Contact

For questions, please contact Yu Xi: yuxi.cs@sjtu.edu.cn or Haoyu Li: haoyu.li.cs@sjtu.edu.cn or open an issue.
