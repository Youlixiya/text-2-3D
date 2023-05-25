# Text-2-3D

# Install

```bash
git clone https://github.com/Youlixiya/text-2-3D.git
cd text-2-3D
```

### Optional: create a python virtual environment

To avoid python package conflicts, we recommend using a virtual environment, e.g.: using conda or venv:

```bash
conda create -n text-2-3D python=3.10
conda activate text-2-3D
```

### Install with pip

```bash
pip install -r requirements.txt
```

### Download pre-trained models

segment-anything

```bash
# move sam pretrained models to pretrained dir
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
We also provide the `setup.py` to build each extension:
```bash
cd text-2-3D

# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
pip install ./raymarching # install to python path (you still need the raymarching/ folder, since this only installs the built extension.)
```

### Taichi backend (optional)
Use [Taichi](https://github.com/taichi-dev/taichi) backend for Instant-NGP. It achieves comparable performance to CUDA implementation while **No CUDA** build is required. Install Taichi with pip:
```bash
pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly
```

# Usage

First time running will take some time to compile the CUDA extensions.

You can refer to [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)

For example commands, check [`scripts`](./scripts).

For advanced tips and other developing stuff, check [Advanced Tips](./assets/advanced.md).

# Acknowledgement

This work is based on an increasing list of amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)

```
@misc{stable-dreamfusion,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/stable-dreamfusion},
    Title = {Stable-dreamfusion: Text-to-3D with Stable-diffusion}
}
```



* [DreamFusion: Text-to-3D using 2D Diffusion](https://dreamfusion3d.github.io/)
    
    ```
    @article{poole2022dreamfusion,
        author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
        title = {DreamFusion: Text-to-3D using 2D Diffusion},
        journal = {arXiv},
        year = {2022},
    }
    ```
    
* [Magic3D: High-Resolution Text-to-3D Content Creation](https://research.nvidia.com/labs/dir/magic3d/)
   ```
   @inproceedings{lin2023magic3d,
      title={Magic3D: High-Resolution Text-to-3D Content Creation},
      author={Lin, Chen-Hsuan and Gao, Jun and Tang, Luming and Takikawa, Towaki and Zeng, Xiaohui and Huang, Xun and Kreis, Karsten and Fidler, Sanja and Liu, Ming-Yu and Lin, Tsung-Yi},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
      year={2023}
    }
   ```

* [Zero-1-to-3: Zero-shot One Image to 3D Object](https://github.com/cvlab-columbia/zero123)
    ```
    @misc{liu2023zero1to3,
        title={Zero-1-to-3: Zero-shot One Image to 3D Object},
        author={Ruoshi Liu and Rundi Wu and Basile Van Hoorick and Pavel Tokmakov and Sergey Zakharov and Carl Vondrick},
        year={2023},
        eprint={2303.11328},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    ```

* [RealFusion: 360° Reconstruction of Any Object from a Single Image](https://github.com/lukemelas/realfusion)
    ```
    @inproceedings{melaskyriazi2023realfusion,
        author = {Melas-Kyriazi, Luke and Rupprecht, Christian and Laina, Iro and Vedaldi, Andrea},
        title = {RealFusion: 360 Reconstruction of Any Object from a Single Image},
        booktitle={CVPR}
        year = {2023},
        url = {https://arxiv.org/abs/2302.10663},
    }
    ```

* [Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation](https://fantasia3d.github.io/)
    ```
    @article{chen2023fantasia3d,
        title={Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation},
        author={Rui Chen and Yongwei Chen and Ningxin Jiao and Kui Jia},
        journal={arXiv preprint arXiv:2303.13873},
        year={2023}
    }
    ```

* [Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior](https://make-it-3d.github.io/)
    ```
    @article{tang2023make,
        title={Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior},
        author={Tang, Junshu and Wang, Tengfei and Zhang, Bo and Zhang, Ting and Yi, Ran and Ma, Lizhuang and Chen, Dong},
        journal={arXiv preprint arXiv:2303.14184},
        year={2023}
    }
    ```

* [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and the [diffusers](https://github.com/huggingface/diffusers) library.

    ```
    @misc{rombach2021highresolution,
        title={High-Resolution Image Synthesis with Latent Diffusion Models},
        author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
        year={2021},
        eprint={2112.10752},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

    @misc{von-platen-etal-2022-diffusers,
        author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
        title = {Diffusers: State-of-the-art diffusion models},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/huggingface/diffusers}}
    }
    ```

* The GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).

* Puppy image from : https://www.pexels.com/photo/high-angle-photo-of-a-corgi-looking-upwards-2664417/

* Anya images from : https://www.goodsmile.info/en/product/13301/POP+UP+PARADE+Anya+Forger.html
