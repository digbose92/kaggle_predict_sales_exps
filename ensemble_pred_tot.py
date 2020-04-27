import pickle 
import pandas as pd 


test  = pd.read_csv('../data/test.csv').set_index('ID')


with open("Pred_check_lgbm.pkl","rb") as f:
   lgbm_pred=pickle.load(f)

with open("Pred_check.pkl","rb") as f:
   pred_vals=pickle.load(f)

#ensemble between lgbm_pred and rf 
rf_pred_val=pred_vals['Y_rf']
lgbm_pred_val=lgbm_pred['Y_lgbm']
xgb_pred_val=pred_vals['Y_xgb']

#ens_pred_1=(0.4*rf_pred_val+0.6*lgbm_pred_val)
#ens_pred_2=(0.6*rf_pred_val+0.4*lgbm_pred_val)
ens_pred_tot=(0.4*rf_pred_val+0.5*lgbm_pred_val+0.1*xgb_pred_val)

"""submission_1 = pd.DataFrame({
    "ID": test.index, 
   "item_cnt_month": ens_pred_1
})"""

submission_2 = pd.DataFrame({
    "ID": test.index, 
   "item_cnt_month": ens_pred_tot
})

#submission_1.to_csv('0.4_rf_0.6_lgbm.csv',index=False)
submission_2.to_csv('0.4_rf_0.5_lgbm_0.1_xgb.csv',index=False)
