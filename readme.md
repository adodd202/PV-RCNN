### Preface
The original creator of the code branched for this project is [Jacob Hultman](https://github.com/jhultman), whose PV-RCNN code can be found [here](https://github.com/jhultman/vision3d). I am also collaborating with my friend, [Dean](dgoldman916), on this project. 

Specifically our goals of this fork are to complete the PV-RCNN implementation begun by JHultman. I have so far built:
- The keypoint weighting module
- Begun the overall loss function (in development)

What's next?
- Overall PV-RCNN loss function
- Verify the ROI Grid Pooling operation
- Implement refinement loss
- Verify the refinement layer
- Verify training works as desired and analyze results
- Build a suite of metrics (FROC, mAP)
- Possibly visualizations

Thank you again to JHultman and the writers of [PV-RCNN](https://arxiv.org/pdf/1912.13192) for providing a starting point to develop this detection code and for providing such a fascinating idea to work on.

# PV-RCNN
An unofficial Pytorch implementation of [PV-RCNN](https://arxiv.org/pdf/1912.13192): Point-Voxel Feature Set Abstraction for 3D Object Detection.

![PV-RCNN](images/pvrcnn.png)

## News (03/02/2020)
- Added implementation of SECOND.

## Project goals
- Emphasis on simple codebase (no 1,000 LOC functions).
- General 3D detection library (easy to extend to new models).
- Hope to reproduce results of paper.

## Status and plans
- This repo is still under active development.
- I will post a pretrained model when codebase stabilizes and results are good.
- I will add more detailed training and inference instructions.
- I will add description of codebase and design choices.

## Usage
See [inference.py](pvrcnn/inference.py).

## Installation
See [install.md](install.md) and please ask if you have any questions. I will supply a Docker build soon.

## Citing
If you find this work helpful in your research, please consider starring this repo and citing:

```
@article{pvrcnnpytorch,
  author={Jacob Hultman},
  title={PV-RCNN PyTorch},
  journal={https://github.com/jhultman/PV-RCNN},
  year={2020}
}
```

and the original PV-RCNN paper (note I am not an author of this paper):

```
@article{shi2019pv,
  author={Shi, Shaoshuai and Guo, Chaoxu and Jiang, Li and Wang, Zhe and Shi, Jianping and Wang, Xiaogang and Li, Hongsheng},
  title={PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection},
  journal={arXiv preprint arXiv:1912.13192},
  year={2019}
}
```

## Contributions
Contributions are welcome. Please post an issue if you find any bugs.

## Acknowledgements and licensing
Please see [license.md](license.md). Note that the code in `pvrcnn/ops` is largely from [detectron2](https://github.com/facebookresearch/detectron2) and hence is subject to the Apache [license](pvrcnn/ops/LICENSE). Thank you to the authors of PV-RCNN for their research.
