import os
import pandas as pd
from sklearn import model_selection

if _name == "__main__":
	input_path = ""
	df = pd.read_csv(os.path.join(input_path,"train.csv"))
	df["kfold"]=-1
	df = df.sample(frac=1).reset_index(drop =True)
	y = df.target.values
	
	# StratifiedKFold takes the cross validation one step further. The class distribution in the dataset is preserved in the training and test splits. 
	kf=model_selection.StratifiedKfold(n_splits=10)
	for fold_,(_,_) in enumerate(kf.split(X=df,y=y)):
		df.loc[:,"kfold"] = fold_
	df.to_csv(os.path.join(input_path,"train_folds.csv"), index=False)
	
	 