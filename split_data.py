from sklearn.model_selection import train_test_split
import pandas as pd

tumor_features = pd.read_csv("brain_tumor_features.csv")
y = tumor_features['tumor_type']
X = tumor_features.drop(columns=['tumor_type'])
# y reprezinta label-ul tumorilor; X reprezinta caracteristicile
# le dam shuffle si le impartim in 80% train - 20% test.
# stratify ne ajuta sa mentinem o impartire uniforma
# sa nu avem 80% din clasa notumor intr-o parte de exemplu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
# salvam fisierele train.csv si test.csv cu datele pe care se va antrena modelul
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)