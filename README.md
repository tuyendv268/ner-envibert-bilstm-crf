## Named Entity Recognition
#### Model Architecture:
  + EnViBERT + BiLSTM + CRF

#### Library:
  + pytorch
  + transformers
  + tqdm
  + numpy
  
#### Data : 
  + VLSP 2016
  
## Kết quả thử nghiệm:  
Entity | precision	 | recall | F1-score
---|---|---|---
`LOC` | 0.88 | 0.89 | 0.88 
`MISC` | 0.88 | 0.90 | 0.89 
`ORG` | 0.70 | 0.73 | 0.72 
`PER` | 0.93 | 0.93 | 0.93
---|---|---|---
`micro avg` | 0.88 | 0.89 | 0.89
`macro avg` | 0.85 | 0.86 | 0.86
`weighted avg` | 0.89 | 0.89 | 0.89