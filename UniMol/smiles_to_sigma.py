from unimol_tools import MolTrain, MolPredict
import numpy as np

model = MolPredict(load_model='sigma_model/new_parameter')


preds = model.predict('SOMAS.csv')

np.savetxt("SOMAS_pred.csv", preds, delimiter=",")