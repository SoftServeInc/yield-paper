# Graph neural networks for efficient yield prediction of chemical reactions
## About
This repo contains supplementary code for our paper *Graph neural networks for efficient yield prediction of chemical reactions* (to be published soon) where we propose to use a graph neural network to predict actual yield of chemical reactions. We evaluate the model performance on three chemical reactions datasets: 2 small scale public single reaction datasets  -- Buchwald-Hartwig(from [Predicting reaction performance in C–N cross-coupling using machine learning](https://science.sciencemag.org/content/360/6385/186.full) and Suzuki-Miyaura reactions (from [A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow](https://science.sciencemag.org/content/359/6374/429) --  as well as one proprietary multiple reaction class dataset. 

We provide preprocessing and atom mapping code for open datasets as well as scripts used for training. Graph neural network code in ```chemprop/``` dir is taken from https://github.com/chemprop/chemprop with some very minor modifications.

We also added a possibility to apply dimensionality reduction(t-sne) to the middle graph representations and the final mixed (graph and rdkit descriptors) representations learned by graph neural net. Example visualizations can be found in ```clustering/``` .
## Installation
Required packages are listed in environment.yml. Just run ```conda env create -f environment.yml ``` 

## Authors
- Dzvenymyra Yarish dyari@softserveinc.com
- Sofiya Garkot sgark@softserveinc.com
- Oleksandr Gurbych 
