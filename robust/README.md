# A Robust and Efficient Framework for Sports-Field Registration

This is an re-implementation of [A Robust and Efficient Framework for Sports-Field Registration](https://openaccess.thecvf.com/content/WACV2021/papers/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.pdf).


## Pretrained Models
1. Download [pretrained weight on WorldCup dataset](https://cgv.cs.nthu.edu.tw/KpSFR_data/model/robust.pth).
2. Now the pretrained models would place in [checkpoints](../checkpoints).
3. Download public [WorldCup](https://nhoma.github.io/) dataset.
4. Now the WorldCup dataset would place in [dataset/soccer_worldcup_2014](../dataset/soccer_worldcup_2014).


## Evaluation
### Evaluation command
```python
python test.py <path/param_text_file>
```
param_text_file as follows,
- `exp_robust.txt`: download [pretrained weight](https://cgv.cs.nthu.edu.tw/KpSFR_data/model/robust.pth) first and place in [checkpoints](../checkpoints). Set `--train_stage` to **0** for testing on WorldCup test set or set `--train_stage` to **1** on TS-WorldCup test set and set `--sfp_finetuned` to **False**.
- `exp_robust_finetuned.txt`: download [pretrained weight](https://cgv.cs.nthu.edu.tw/KpSFR_data/model/robust_finetuned.pth) first and place in [checkpoints](../checkpoints). Set `--train_stage` to **1** and `--sfp_finetuned` to **True** for testing finetuned results on TS-WorldCup test set.

We will save heatmap results and corresponding homography matrix into **/checkpoints/path of experimental name**, which set `--name` in param_text_file.

Note:
- `robust_worldcup_testset_dilated`: the preprocess results for predicting on WorldCup dataset and would place in [dataset/soccer_worldcup_2014/soccer_data](../dataset/soccer_worldcup_2014/soccer_data).
- `SingleFramePredict_with_normalized`: the preprocess results for predicting on TS-WorldCup dataset and would place in [dataset/WorldCup_2014_2018](../dataset/WorldCup_2014_2018).
- `SingleFramePredict_finetuned_with_normalized`: the preprocess results for finetuning on TS-WorldCup dataset and would place in [dataset/WorldCup_2014_2018](../dataset/WorldCup_2014_2018).


## Train model
### Train command
```python
python train.py <path/param_text_file>
```
param_text_file as follows,
- `opt_robust.txt`: download [pretrained weight](https://cgv.cs.nthu.edu.tw/KpSFR_data/model/robust.pth) first and place in [checkpoints](../checkpoints). Set `--train_stage` to **0** and `--trainset` to **train_val** for training on WorldCup train set. 
- `opt_robust_finetuned.txt`: download [pretrained weight](https://cgv.cs.nthu.edu.tw/KpSFR_data/model/robust_finetuned.pth) first and place in [checkpoints](../checkpoints). Set `--train_stage` to **1** and `--trainset` to **train** for finetuning on WorldCup train set.

We will save visualize results and weights into **/checkpoints/path of experimental name**, which set `--name` in param_text_file.

Note: Please check the following arguments to set correct before training every time.
- `--gpu_ids`
- `--name`
- `--train_stage` and `--trainset`
- `--ckpt_path`
- `--train_epochs` and `--step_size`

Details refer to [options.py](../options.py).