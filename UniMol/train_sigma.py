from unimol_tools import MolTrain, MolPredict

model = MolTrain(task='multilabel_regression', # multilabel_regression, regression
                data_type='molecule',
                epochs=100, # 50
                learning_rate=0.0001, # 0.0001
                batch_size=8,
                early_stopping=10, # 5
                metrics='mse',
                split='random',
                save_path='/lipo/self_nohs_model/',
                remove_hs=True, # False
                config='sigma48.yaml'
              )

model.fit('/lipo/lipo_feats_log.csv')