# Graph neural networks for efficient yield prediction of chemical reactions
## About
This repo contains supplementary code for our paper *Graph neural networks for efficient yield prediction of chemical reactions* 
(to be published soon) where we propose to use a graph neural network to predict actual yield of chemical reactions. 
We evaluated the model performance on three chemical reactions datasets: 2 small scale public single reaction datasets  
-- Buchwald-Hartwig (from [Predicting reaction performance in C–N cross-coupling using machine learning](https://science.sciencemag.org/content/360/6385/186.full) and Suzuki-Miyaura reactions (from [A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow](https://science.sciencemag.org/content/359/6374/429) 
--  as well as one proprietary multiple reaction class dataset, provided by Enamine. We conducted a detailed analysis
of model's errors on the commercial dataset and provided a chemically viable explanation for the most common of them.

We provide preprocessing and atom mapping code for open datasets as well as scripts used for training. Graph neural network code in ```chemprop/``` dir is taken from https://github.com/chemprop/chemprop with some very minor modifications.
For example, we added a possibility to apply dimensionality reduction(t-sne) to the middle graph representations and the final mixed (graph and rdkit descriptors) representations learned by graph neural net. Example visualizations can be found in ```clustering/``` .

## Installation
Required packages are listed in environment.yml. Just run ```conda env create -f environment.yml ``` 

## To reproduce
For single reaction class datasets:
1. Run ```single_reaction_class_data_preprocessing.ipynb```
2. Run ``` bash train_k_fold.sh```

## Authors
- Dzvenymyra Yarish    `dyari@softserveinc.com`
- Sofiya Garkot        `sgark@softserveinc.com`
- Oleksandr Grygorenko `o.grygorenko@gmail.com`
- Yurii Moroz          `ysmoroz@gmail.com`
- Dmytro Radchenko     `d.radchenko@enamine.net`
- Oleksandr Gurbych    `oleksandr.v.hurbych@lpnu.ua`
