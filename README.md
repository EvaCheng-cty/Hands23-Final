# Hand Object Detector 
Detectron2 implementation for the paper, *Understanding Human Hands in Contact at Internet Scale* (CVPR 2020, **Oral**).

### Environment

- Set up detectron2 environment as in [install.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

### Train

```
CUDA_VISIBLE_DEVICES=4 python trainval_net.py --num-gpus 1 --config-file faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml
```

### Evaluation

```
CUDA_VISIBLE_DEVICES=4 python trainval_net.py --num-gpus 1 --config-file faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml --eval-only MODEL.WEIGHTS path/to/model.pth
```

### Demo

## Citation

If this work is helpful in your research, please cite:
```
@INPROCEEDINGS{Shan20, 
    author = {Shan, Dandan and Geng, Jiaqi and Shu, Michelle  and Fouhey, David},
    title = {Understanding Human Hands in Contact at Internet Scale},
    booktitle = CVPR, 
    year = {2020} 
}
```
When you use the model trained on our ego data, make sure to also cite the original datasets ([Epic-Kitchens](https://epic-kitchens.github.io/2018), [EGTEA](http://cbs.ic.gatech.edu/fpv/) and [CharadesEgo](https://prior.allenai.org/projects/charades-ego)) that we collect from and agree to the original conditions for using that data.