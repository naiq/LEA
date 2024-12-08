# LEA: Learning Latent Embedding Alignment Model for fMRI Decoding and Encoding
Pytorch implementation of 'Learning Latent Embedding Alignment Model for fMRI Decoding and Encoding' In TMLR, 2024
<p align="center">
<img src=assets/teaser.jpg />
</p>


## Abstract
The connection between brain activity and visual stimuli is crucial to understanding the human brain. Although deep generative models have shown advances in recovering brain recordings by generating images conditioned on fMRI signals, it is still challenging to generate consistent semantics. Moreover, predicting fMRI signals from visual stimuli remains a hard problem. In this paper, we introduce a unified framework that addresses both fMRI decoding and encoding. We train two latent spaces to represent and reconstruct fMRI signals and visual images, respectively. By aligning these two latent spaces, we seamlessly transform between the fMRI signal and visual stimuli. Our model, called Latent Embedding Alignment (LEA), can recover visual stimuli from fMRI signals and predict brain activity from images. LEA outperforms existing methods on multiple benchmark fMRI decoding and encoding datasets. It offers a comprehensive solution for modeling the relationship between fMRI signals and visual stimuli. 

## Framework
<p align="center">
<img src=assets/framework />
</p>
- LEA encompasses two distinct encoder-decoder architectures for handling fMRI signals and visual stimuli. 
- An alignment module is further introduced to facilitate the transformation between the latent representations of fMRI signals and visual stimuli.
