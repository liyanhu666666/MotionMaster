# MotionMaster: Generalizable Text-Driven Motion Generation and Editing

This is the code repository of **MotionMaster: Generalizable Text-Driven Motion Generation and Editing** at **CVPR 2026**.

📝 [**arXiv**](https://arxiv.org) | 🌐 [**Project Page**](https://jnnan.github.io/motionmaster/)

# Getting Started

### Prerequisites
- Python 3.10
- CUDA-capable GPU
- Required Python packages (see Installation)

### Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/liyanhu666666/MotionMaster.git
    cd MotionMaster
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
    pip install -e src/human_body_prior_repo/
    ```

3. **Download Checkpoints**:

    Download `mllm/`, `tokenizer.pt`, and `norm_stats.npz` from [Google Drive](https://drive.google.com/drive/folders/1gEQggeBceVOG0CdyZZmIb2uBBV-ux8-q?usp=drive_link) and place them as below.

    The SMPL-X body model (`SMPLX_MALE.npz`) and VPoser weights (`V02_05/`) must be downloaded separately from the [SMPL-X project page](https://smpl-x.is.tue.mpg.de) and placed at the paths shown below:

    ```plaintext
    checkpoints/
    ├── mllm/                   - fine-tuned Qwen2.5-VL language model
    ├── tokenizer.pt            - FSQ motion tokenizer weights
    ├── norm_stats.npz          - normalization statistics
    └── smplx_model/
        └── SMPLX_MALE.npz     - SMPL-X body model (download from SMPL-X project page)
    src/human_body_prior_repo/support_data/dowloads/
    └── V02_05/                 - VPoser model (download from SMPL-X project page)
    ```

### Project Structure

```plaintext
MotionMaster/
├── src/
│   ├── models/                - FSQ tokenizer model
│   ├── quantizers/            - FSQ quantizer
│   ├── smplx_fast/            - SMPL-X (https://github.com/vchoutas/smplx.git)
│   ├── human_body_prior_repo/ - VPoser (https://github.com/nghorbani/human_body_prior.git)
│   ├── feature_utils.py       - motion feature utilities
│   ├── joint2smplx.py         - 3D joints → SMPL-X fitting
│   └── smplx2joints.py        - forward kinematics
├── checkpoints/               - model weights (download separately)
├── infer.py                   - inference entry point
└── README.md
```

### Running Inference

```sh
python infer.py --text "a person walks forward" --output output.pkl
```

Additional arguments:
```
--text          Text description of the motion (required)
--output        Output .pkl file path (default: output.pkl)
--mllm_path     Path to MLLM checkpoint (default: checkpoints/mllm)
--tokenizer_pt  Path to FSQ tokenizer (default: checkpoints/tokenizer.pt)
--stats_npz     Path to normalization stats (default: checkpoints/norm_stats.npz)
--smplx_path    Path to SMPL-X model (default: checkpoints/smplx_model)
```

### Output Format

The output `.pkl` file contains SMPL-X parameters:
```plaintext
{
    "body_pose":      numpy.ndarray (N, 63),
    "global_orient":  numpy.ndarray (N, 3),
    "transl":         numpy.ndarray (N, 3),
}
```

# Citation

```plaintext
@inproceedings{jiang2026motionmaster,
  title={MotionMaster: Generalizable Text-Driven Motion Generation and Editing},
  author={Jiang, Nan and Li, Yunhao and Pang, Lexi and He, Zimo and Huang, Siyuan and Zhu, Yixin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

