## Moe-VRD

This is the source code for the Moe-VRD Project as maintained by the VIP lab at the University of Waterloo. This code creates a mixture of experts framework and encapsulates the work done by Shang et al.'s [VidVRD-II](https://xdshang.github.io/assets/pdf/VidVRD-II-preprint.pdf) to a singular expert, which can then be replicated and scaled accordingly. 

--

**Note**
This work is in progress, and as this project is relatively new on Github code-wise there will be lots of changes over time.

--


### Environment
The setup is very similar to Shang et al.'s code setup:

1. Download [ImageNet-VidVRD dataset](https://xdshang.github.io/docs/imagenet-vidvrd.html) and [VidOR dataset](https://xdshang.github.io/docs/vidor.html). Then, place the data under the same parent folder as this repository.
```
2. Install dependencies (tested with TITAN Xp GPU, Nvidia RTX A6000)
<!-- Ubuntu 18.04.5 LTS (GNU/Linux 4.15.0-142-generic x86_64); NVIDIA Driver Version: 460.73.01,  -->
```
conda create -n moe-vrd -c conda-forge python=3.7 Cython tqdm scipy "h5py>=2.9=mpi*" ffmpeg=3.4 cudatoolkit=10.1 cudnn "pytorch>=1.7.0=cuda101*" "tensorflow>=2.0.0=gpu*"
conda activate moe-vrd
python setup.py build_ext --inplace
``` 

### Quick Start
1. Download the precomputed object tracklets and features for ImageNet-VidVRD ([437MB](https://zdtnag7mmr.larksuite.com/file/boxusOKDuonKBcA2Le9JMeu3ePg)) and VidOR (32GB: [part1](https://zdtnag7mmr.larksuite.com/file/boxusff3nUWfZly119sJXhLkpWe), [part2](https://zdtnag7mmr.larksuite.com/file/boxus2FJ47VRRNSUZ4f9aJzICAc), [part3](https://zdtnag7mmr.larksuite.com/file/boxusr0Bb6cSX4OQXCjKvhHf94d), [part4](https://zdtnag7mmr.larksuite.com/file/boxusGIpO3VDPkfNTtzUqMRIdrc)), and extract them under `imagenet-vidvrd-baseline-output` and `vidor-baseline-output` as above, respectively.
2. Run `python main.py --cfg config/imagenet_vidvrd_3step_prop_wd0.01.json --id 3step_prop_wd0.01 --train --cuda` to train the model for ImageNet-VidVRD. Use `--cfg config/vidor_3step_prop_wd1.json` for VidOR.
3. Run `python main.py --cfg config/imagenet_vidvrd_3step_prop_wd0.01.json --id 3step_prop_wd0.01 --detect --cuda` to detect video relations (inference) and the results will be output to `../imagenet-vidvrd-baseline-output/models/3step_prop_wd0.01/video_relations.json`.
4. Run `python evaluate.py imagenet-vidvrd test relation ../imagenet-vidvrd-baseline-output/models/3step_prop_wd0.01/video_relations.json` to evaluate the results.
5. To visualize the results, add the option `--visualize` to the above command (this will involve `visualize.py` so please make sure the environment is switched according to the last section). For the better visualization mentioned in the paper, change `association_algorithm` to `graph` in the configuration json, and then run Step 3 and 5.
6. To automatically run the whole traininng and test pipepine multiple times, run `python main.py --cfg config/imagenet_vidvrd_3step_prop_wd0.01.json --id 3step_prop_wd0.01 --pipeline 5 --cuda --no_cache` and then you can obtain a mean/std result.

### Object Tracklet Extraction (optional)
1. We extract frame-level object proposals using the off-the-shelf tool. Please first download and install [tensorflow model library](https://github.com/tensorflow/models/tree/master/research/object_detection). Then, run `python -m video_object_detection.tfmodel_image_detection [imagenet-vidvrd/vidor] [train/test/training/validation]`. You can also download our precomputed results for ImageNet-VidVRD ([6GB](https://zdtnag7mmr.larksuite.com/file/boxuspVnSh0mnRW4Zdxh3282w4d)).
2. To obtain object tracklets based on the frame-level proposals, run `python -m video_object_detection.object_tracklet_proposal [imagenet-vidvrd/vidor] [train/test/training/validation]`.

### Acknowledgement
This repository is built based on [VidVRD-helper](https://github.com/xdshang/VidVRD-helper) and [VidVRD-II](https://github.com/xdshang/VidVRD-II). If this repo is helpful in your research, you can use the following bibtex to cite their paper:
```
@inproceedings{shang2021video,
    author={Shang, Xindi and Li, Yicong and Xiao, Junbin and Ji, Wei and Chua, Tat-Seng},
    title={Video Visual Relation Detection via Iterative Inference},
    booktitle={ACM International Conference on Multimedia},
    year={2021}
}
```
