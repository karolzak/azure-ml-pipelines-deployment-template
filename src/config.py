model_name = 'diabetes_model'

# train 
test_size = 0.2
random_state = 66
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
model_name_base = 'model_alpha_%s.pkl'
feat_cols = ['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10']
label_col = 'level'