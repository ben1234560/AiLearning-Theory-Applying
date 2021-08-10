import joblib
cross_le = joblib.load('/data/didi_2021/model_h5/crossid_le')
print(len(cross_le.classes_.tolist()))
