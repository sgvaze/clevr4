# Clevr-4
This repo contains a starter Jupyter Notebook and utilities for the Clevr-4 dataset.

![image](assets/dataset.png)

## Summary

Clevr-4 is a synthetic dataset where image generation is conditioned on four attributes, based on object: *texture*, *shape*, *color*, *count*.

As a result, the dataset contains four equally valid clusterings of the same set of images. Each clustering (a.k.a 'taxonomy') has 10 categories (i.e 10 possible values for each of texture, shape, color and count.)

It can be used as a probing set for biases in representation learning and for category discovery.

More information is provided on the webpage [here](https://www.robots.ox.ac.uk/~vgg/data/clevr4/).

## Download

There are two versions of the dataset, one with [10k images](https://thor.robots.ox.ac.uk/clevr4/clevr_4_10k_v1.zip) (used in the original paper) and one with [100k images](https://thor.robots.ox.ac.uk/clevr4/clevr_4_109k_v1.zip). The 10k set contains some class imbalance. Follow the hyperlinks to download the images, or else the following instructions:

```
# Download 10k and 100k datasets
cd PATH/TO/CLEVR4
wget https://thor.robots.ox.ac.uk/clevr4/clevr_4_10k_v1.zip     # 10k dataset
wget https://thor.robots.ox.ac.uk/clevr4/clevr_4_100k_v1.zip    # 100k dataset
wget https://thor.robots.ox.ac.uk/clevr4/SHA512SUMS

# Check the integrity of the downloads
sha512sum -c SHA512SUMS
```

Now unzip as:

```
unzip clevr_4_10k_v1.zip -d clevr_4_10k
unzip clevr_4_100k_v1.zip -d clevr_4_100k
```

## Running the starter notebook

To run the notebook, you will need the following through pip/conda:

```
torch
torchvision
faiss-gpu       # For fast k-means clustering
scikit-learn    # For metric computation

jupyter
matplotlib
```



## Citation

If you use this code in your research, please consider citing our paper:
```
@InProceedings{vaze2023clevr4,
               title={No Representation Rules Them All in Category Discovery},
               author={Sagar Vaze and Andrea Vedaldi and Andrew Zisserman},
               booktitle={Advances in Neural Information Processing Systems 37},
               year={2023}
}
```

