import pandas as pd
import numpy as np
import pdb
pdb.set_trace()
test_result = pd.read_csv("test_result.csv", header=None)
test_original = pd.read_csv("test_original.csv", header=None)

n = test_original.shape[0]

rmse_c1 = np.sqrt(1/(n**2) * np.sum(((test_result.iloc[:,0]-test_original.iloc[:,0])/test_original.iloc[:,0])**2))
rmse_c2 = np.sqrt(1/(n**2) * np.sum(((test_result.iloc[:,1]-test_original.iloc[:,1])/test_original.iloc[:,1])**2))
rmse_c3 = np.sqrt(1/(n**2) * np.sum(((test_result.iloc[:,2]-test_original.iloc[:,2])/test_original.iloc[:,2])**2))
rmse_c4 = np.sqrt(1/(n**2) * np.sum(((test_result.iloc[:,3]-test_original.iloc[:,3])/test_original.iloc[:,3])**2))
rmse_u = np.sqrt(1/(n**2) * np.sum(((test_result.iloc[:,4]-test_original.iloc[:,4])/test_original.iloc[:,4])**2))
rmse_h = np.sqrt(1/(n**2) * np.sum(((test_result.iloc[:,5]-test_original.iloc[:,5])/test_original.iloc[:,5])**2))
rmse_t = np.sqrt(1/(n**2) * np.sum(((test_result.iloc[:,6]-test_original.iloc[:,6])/test_original.iloc[:,6])**2))

bias_c1 = 1/n * np.sum(test_result.iloc[:,0]-test_original.iloc[:,0])
bias_c2 = 1/n * np.sum(test_result.iloc[:,1]-test_original.iloc[:,1])
bias_c3 = 1/n * np.sum(test_result.iloc[:,2]-test_original.iloc[:,2])
bias_c4 = 1/n * np.sum(test_result.iloc[:,3]-test_original.iloc[:,3])
bias_u = 1/n * np.sum(test_result.iloc[:,4]-test_original.iloc[:,4])
bias_h = 1/n * np.sum(test_result.iloc[:,5]-test_original.iloc[:,5])
bias_t = 1/n * np.sum(test_result.iloc[:,6]-test_original.iloc[:,6])

data = [rmse_c1, rmse_c2, rmse_c3, rmse_c4, rmse_u, rmse_h, rmse_t,\
        bias_c1,bias_c2, bias_c3, bias_c4, bias_u, bias_h, bias_t]

index_label = ['rmse_c1','rmse_c2','rmse_c3', 'rmse_c4','rmse_u','rmse_h','rmse_t',\
        "bias_c1","bias_c2",'bias_c3','bias_c4','bias_u',"bias_h",'bias_t']
df = pd.DataFrame(data, index=index_label)
df.to_csv('statistical_data_2.csv', header=None)
