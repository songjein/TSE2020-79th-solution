## 2020 Tweet Setiment Extraction 79th solution 
final LB score: .718

## Training
- ./script/start_train 

## pseudo labeling
1. filter sentiment 140 by using ./notebooks/mw_sent_aug.ipynb
    - high confidence + text blob score 
2. ./scripts/start_pseudo
3. ./scripts/start_train (args.use_pseudo_label)
