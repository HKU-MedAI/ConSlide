# ConSlide
[ICCV 2023] ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis.

## Training Data Preparation

We mainly follow the pipeline of [CLAM](https://github.com/mahmoodlab/CLAM). The modified version of the CLAM code for data preparation will be released later.

## Training Example

```
python utils/main.py --state train --model conslide --dataset seq-wsi --exp_desc conslide --buffer_size 1100 --alpha 0.2 --beta 0.2
```

## Updates / TODOs
Please follow this GitHub for more updates.

- [ ] Refine the code.
- [ ] Provide code for data preparation.
- [ ] Remove dead code.
- [ ] Better documentation on interpretability code example.

## Reference
If you find our work useful in your research please consider citing our [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_ConSlide_Asynchronous_Hierarchical_Interaction_Transformer_with_Breakup-Reorganize_Rehearsal_for_Continual_ICCV_2023_paper.html):

Huang, Y., Zhao, W., Wang, S., Fu, Y., Jiang, Y., & Yu, L. (2023). ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 21349-21360).

```
@inproceedings{huang2023conslide,
  title={ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis},
  author={Huang, Yanyan and Zhao, Weiqin and Wang, Shujun and Fu, Yu and Jiang, Yuming and Yu, Lequan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21349--21360},
  year={2023}
}
```

## Acknowledgements

Framework code for Continual Learning was largely adapted via making modifications to [Mammoth](https://github.com/aimagelab/mammoth)