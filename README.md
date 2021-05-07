# Computing FID and IS with paths to image folders


### Prerequisites
[torch](http://pytorch.org/), [torchvision](https://github.com/pytorch/vision), [numpy/scipy](https://scipy.org/).

### Examples
Computing FID of two given folders containing images:

```
python fid_score.py --path1 /path_to_folder_1 --path2 /path_to_folder_2
```

Computing IS of a given folder containing images:
```
python inception_score.py --path /path_to_folder_1
```
