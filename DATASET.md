Public Soccer WorldCup ([Download](https://nhoma.github.io/))
- Train Set Path: dataset/soccer_worldcup_2014/soccer_data/train_val
- Test Set Path: dataset/soccer_worldcup_2014/soccer_data/test

Details refer to [source](dataset/soccer_worldcup_2014/soccer_data/source.txt).

TS-WorldCup ([Download](https://cgv.cs.nthu.edu.tw/KpSFR_data/TS-WorldCup.zip))
- Train Set Path: 
    - RGB: dataset/WorldCup_2014_2018/Dataset/80_95/ + train.txt
    - GT: dataset/WorldCup_2014_2018/Annotations/80_95/ + train.txt

- Test Set Path: 
    - RGB: dataset/WorldCup_2014_2018/Dataset/80_95/ + test.txt
    - GT: dataset/WorldCup_2014_2018/Annotations/80_95/ + test.txt

- Misc
    - SingleFramePredict_with_normalized/SingleFramePredict_finetuned_with_normalized: the prediction of Nie et al.

We would predict keypoints label based on the heatmap results of Nie et al.