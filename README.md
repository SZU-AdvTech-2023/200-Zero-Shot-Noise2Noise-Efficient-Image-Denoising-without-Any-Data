# Resources

- [Arxiv](https://arxiv.org/pdf/2101.02824.pdf)
- [Conference](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.pdf)
- [Supplementary](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_CVPR_2021_supplemental.pdf)

# Python Requirements

This code was tested on:

- Python 3.7
- Pytorch 1.7

# Training without using ground-truth data

run

```
python main.py -i {noisy_image_path}
```

The clean image will be generated under `/output`.