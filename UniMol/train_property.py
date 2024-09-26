from unimol_tools import MolTrain, MolPredict

model = MolTrain(task='regression', # multilabel_regression, regression, classification
                data_type='molecule',
                epochs=100, # 50, 100
                learning_rate=0.0001, # 0.0001
                batch_size=8, # 8
                early_stopping=10, # 5, 10
                metrics='mse', # mse, auc
                split='random', # random
                save_path='/lipo/self_nohs_model/property/',
                remove_hs=True, # True, False
                config='sigma48.yaml'
              )


model.fit('/MoleculeNet/unimol_lipo.csv')