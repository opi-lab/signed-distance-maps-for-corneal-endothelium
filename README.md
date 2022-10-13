# Corneal endothelium assessment in specular microscopy images with Fuchs' dystrophy via deep regression of signed distance maps

#### by Juan S. Sierra et al. (https://www.researchgate.net/profile/Juan-Sierra-Bravo) 

The original paper can be found  [here](https://www.researchgate.net/profile/Juan-Sierra-Bravo).

## Abstract
Specular microscopy assessment of the human corneal endothelium (CE) in Fuchs' dystrophy is challenging due to the presence of dark image regions called guttae. This paper proposes a UNet-based segmentation approach that requires minimal post-processing and achieves reliable CE morphometric assessment and guttae identification across all degrees of Fuchs' dystrophy. We cast the segmentation problem as a regression task of the cell and gutta signed distance maps instead of a pixel-level classification task as typically done with UNets. Compared to the conventional UNet classification approach, the distance-map regression approach converges faster in clinically relevant parameters. It also produces morphometric parameters that agree with the manually-segmented ground-truth data, namely the average cell density difference of -41.9 cells/mm<sup>2</sup> (95% confidence interval (CI) [-306.2, 222.5]) and the average difference of mean cell area of 14.8 μm<sup>2</sup> (95% CI [-41.9, 71.5]). These results suggest a promising alternative for CE assessment. 

## About this repository

### Files

This repository is a simplified version of the project, it doesn't contain the postprocessing operations.

* Clone this repository: `git clone https://github.com/opi-lab/signed-distance-maps-for-corneal-endothelium.git`
* Download the model [here](https://drive.google.com/drive/folders/).
* Move the `model/` folder to the datasets folder: `mv path/to/model/ datasets/`.

The folder structure should be:

```
signed-distance-maps-for-corneal-endothelium/
    ├── datasets/
        ├── model/
        ├── training/
        └── validation/
    ├── guttae/
        ├── deeptrack/
        ├── loaders.py
        ├── models.py
        └── utils.py
    ├── imgs/
    ├── seg_data.ipynb
    ├── setup.py
    ├── test_dataset.py
    └── train.py
```

### Try the trained model

In the directory `./datasets/validation/` of the main folder, there is an image of cornea guttata and its curated segmentation. To run the trained model, check the `./seg_data.ipynb` notebook.

### Train the model

To train the model from console, you must use the command:

`python train.py setup [-i | -e | -p | -r | -n]`

<strong>Options</strong>

`-i` Index. It is possible to define more than one architecture in the `./setup.py` file. The index defines which of the architectures will be trained.

`-e` Epochs. The default value is $100'000$. It defines the number of epochs.

`-p` Patience. The default value is $100$. Number of epochs with no improvement after which training will be stopped.

`-r` Repetitions. The default value is $1$. Number of times the model will be trained.

`-n` Model name. The default value is the name of the computer username. After finishing the training, the models and the weights will be saved with the given name. 

<strong>Example</strong>

`python train.py setup -i 0 -e 10 -r 10 -n test`

This command will execute the training of the first architecture 10 times with 10 epochs, after which the model will be saved with the name <em>test</em>. So, when the training has finished, 10 trained models will be saved in the directory `./datasets/models/test/`.

<!--
<p align="center">
    <img src="imgs/model.png" width="600px"/>
</p>
<p align="center">
    <b>Fig. 1 </b> CNN architecture.
</p>
-->

<!--
## Citation

If you find this code useful, please consider citing:

```
bibtext
```
-->
