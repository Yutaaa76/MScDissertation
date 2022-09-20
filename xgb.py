from xgboost import XGBClassifier

import numpy as np
import pandas as pd
from handle_data import get_text_data

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

train_set, val_set, test_set = get_text_data.get_data()
print(len(train_set), len(val_set), len(test_set))
# print(train_set.columns)

train_data = np.stack(train_set.text_emb)
val_data = np.stack(val_set.text_emb)
test_data = np.stack(test_set.text_emb)

# train_data = np.hstack((np.stack(train_set.text_emb), np.stack(train_set.num_retweets)))
# val_data = np.hstack((np.stack(val_set.text_emb), np.stack(val_set.num_retweets)))
# test_data = np.hstack((np.stack(test_set.text_emb), np.stack(test_set.num_retweets)))

train_label = train_set['label'].to_numpy()
val_label = val_set['label'].to_numpy()
test_label = test_set['label'].to_numpy()

model_1 = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1)
model_2 = XGBClassifier(n_estimators=500, n_jobs=-1)
model_3 = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)

voter = VotingClassifier(estimators=[('rf', model_1), ('xgb', model_2), ('et', model_3)],
                         voting='hard')
voter = voter.fit(train_data, train_label)

train_pred = voter.predict(train_data)
val_pred = voter.predict(val_data)
test_pred = voter.predict(test_data)

print(balanced_accuracy_score(train_label, train_pred), balanced_accuracy_score(val_label, val_pred), balanced_accuracy_score(test_label, test_pred))

print(f1_score(train_label, train_pred, average='macro'), f1_score(val_label, val_pred, average='macro'), f1_score(test_label, test_pred, average='macro'))

