import scipy.io
import numpy as np
import math
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

Train_data_0= scipy.io.loadmat('train_0_img.mat')
Test_data_0=scipy.io.loadmat('test_0_img.mat')
Train_data_0=Train_data_0['target_img']
Test_data_0=Test_data_0['target_img']
Train_data_1= scipy.io.loadmat('train_1_img.mat')
Test_data_1=scipy.io.loadmat('test_1_img.mat')
Train_data_1=Train_data_1['target_img']
Test_data_1=Test_data_1['target_img']

train_mat_n = np.zeros((2,12665))

###############################
# Training set 1
###############################
#print(Train_data_1) to understand the data_structure
#feature 1
Train_data_1_reshaped=Train_data_1.reshape(784,6742)
pixels_sum=[ sum(row[i] for row in Train_data_1_reshaped) for i in range(len(Train_data_1_reshaped[0])) ]
avg_pixels_sum= [x /784 for x in pixels_sum]
#feature 2

avg_var_sum=[]
for i in range(0,6742):
	var_data = Train_data_1_reshaped[:,i].reshape(28,28)
	var_x = np.var(var_data,axis=1)
	var_sum = np.sum(var_x)
	avg_var_sum.append(var_sum /28)
# avg_var_sum = np.asarray(avg_var_sum)


# var=[]
# for i in Train_data_1:
#     var.append(np.var(i,axis=0))
#     # print(i.shape)
# # print(len(var[0]))
# var_sum=[sum(row[i] for row in var) for i in range(len(var[0]))]
# # print(len(var_sum))
# avg_var_sum= [x /28 for x in var_sum]
# print(avg_var_sum.shape)

n = 6742
train_mat_1 = np.zeros((2,n))
for i in range(0,n):
	train_mat_n[0][i] = avg_pixels_sum[i]
	train_mat_n[1][i] = avg_var_sum[i]
	train_mat_1[0][i] = avg_pixels_sum[i]
	train_mat_1[1][i] = avg_var_sum[i]

##################################
#Parameter estimation for 1
##################################
u_mat_1 = np.mean(train_mat_1, axis = 1)
u_mat_1 = u_mat_1.reshape((2,1))
 

x= np.zeros((2,2))
for i in range(0,n):
	x += (train_mat_1[:,i].reshape((2,1)) - u_mat_1) @ (train_mat_1[:,i].reshape((2,1)) - u_mat_1).T

# print(train_mat_1[:, 0])

sigma_1 = x/n
sigma_1[0][1] =0
sigma_1[1][0] =0
print("Parametric estimation of sample 1: ")
print(sigma_1)

###############################
# Training set 0
###############################
n = 5923
Train_data_0_reshaped=Train_data_0.reshape(784,n)
pixels_sum=[ sum(row[i] for row in Train_data_0_reshaped) for i in range(len(Train_data_0_reshaped[0])) ]
avg_pixels_sum= [x /784 for x in pixels_sum]
# print(avg_pixels_sum[0])

# for i in Train_data_0:
#     var.append(np.var(i,axis=0))
# var_sum=[sum(row[i] for row in var) for i in range(len(var[0]))]
# avg_var_sum= [x /28 for x in var_sum]
avg_var_sum=[]
for i in range(0,n):
	var_data = Train_data_0_reshaped[:,i].reshape(28,28)
	var_x = np.var(var_data,axis=1)
	var_sum = np.sum(var_x)
	avg_var_sum.append(var_sum /28)



train_mat_0 = np.zeros((2,n))
for i in range(0,n):
	train_mat_n[0][i+6742] = avg_pixels_sum[i]
	train_mat_n[1][i+6742] = avg_var_sum[i]
	train_mat_0[0][i] = avg_pixels_sum[i]
	train_mat_0[1][i] = avg_var_sum[i]

##################################
#Parameter estimation for 0
##################################
u_mat_0 = np.mean(train_mat_0, axis = 1)
u_mat_0 = u_mat_0.reshape((2,1))
 
x= np.zeros((2,2))
for i in range(0,n):
	x += (train_mat_0[:,i].reshape(2,1) - u_mat_0) @ (train_mat_0[:,i].reshape(2,1) - u_mat_0).T


sigma_0 = x/n
sigma_0[0][1] =0
sigma_0[1][0] =0
print("Parametric estimation of sample 0:")
print(sigma_0)

# ################################
# combined_training = np.zeros((2, (6742+5923)))
# combined_training[0] = np.concatenate((train_mat_0[0], train_mat_1[0]), axis=None)
# combined_training[1] = np.concatenate((train_mat_0[1], train_mat_1[1]), axis=None)
# c_t_transpose = combined_training.T
# print(c_t_transpose.shape)

# y = np.zeros((12665, 1))
# for i in range(0, 5923):
# 	y[i] = 0
# for i in range(5923, 12665):
# 	y[i] = 1

# model = SVC(kernel='rbf')
# model.fit(c_t_transpose, y)

################################
# Creating test matrix
###############################
n = 1135
Test_data_1_reshaped=Test_data_1.reshape(784, n)
pixels_sum=[ sum(row[i] for row in Test_data_1_reshaped) for i in range(len(Test_data_1_reshaped[0])) ]
avg_pixels_sum= [x /784 for x in pixels_sum]

avg_var_sum=[]
for i in range(0,n):
	var_data = Test_data_1_reshaped[:,i].reshape(28,28)
	var_x = np.var(var_data,axis=1)
	var_sum = np.sum(var_x)
	avg_var_sum.append(var_sum /28)


test_mat = np.zeros((2,2115))
for i in range(0,n):
	test_mat[0][i] = avg_pixels_sum[i]
	test_mat[1][i] = avg_var_sum[i]


#### 0
n = 980
Test_data_0_reshaped=Test_data_0.reshape(784, n)
pixels_sum=[ sum(row[i] for row in Test_data_0_reshaped) for i in range(len(Test_data_0_reshaped[0])) ]
avg_pixels_sum= [x /784 for x in pixels_sum]

# var=[]
# for i in Test_data_0:
#     var.append(np.var(i,axis=0))
# var_sum=[sum(row[i] for row in var) for i in range(len(var[0]))]
# avg_var_sum= [x /28 for x in var_sum]

avg_var_sum=[]
for i in range(0,n):
	var_data = Test_data_0_reshaped[:,i].reshape(28,28)
	var_x = np.var(var_data,axis=1)
	var_sum = np.sum(var_x)
	avg_var_sum.append(var_sum /28)

j = 0
for i in range(1135,2115):
	test_mat[0][i] = avg_pixels_sum[j]
	test_mat[1][i] = avg_var_sum[j]
	j+=1

# t_m_transpose = test_mat.T
# y_out = model.predict(t_m_transpose)
# y_test = np.zeros((2115, 1))
# for i in range(0, 1135):
# 	y_test[i] = 1
# for i in range(1135, 2115):
# 	y_test[i] = 0
# print(accuracy_score(y_out, y_test))

################################
# Classifier
################################

sigma_det_1 = np.linalg.det(sigma_1)
sigma_det_0 = np.linalg.det(sigma_0)
result = []
for i in range(0,2115):

	x = ((test_mat[:,i].reshape(2,1) - u_mat_1).T @ np.linalg.inv(sigma_1)) @ (test_mat[:,i].reshape(2,1) - u_mat_1)

	likelihood_1 = 1/math.sqrt(4 * (math.pi**2) * sigma_det_1) \
	* math.exp(-0.5 * x)
 

	likelihood_0 = 1/math.sqrt(4 * (math.pi**2) * sigma_det_0) \
	*	math.exp(-0.5 * (test_mat[:,i].reshape(2,1) - u_mat_0).T @ np.linalg.inv(sigma_0) @ (test_mat[:,i].reshape(2,1) - u_mat_0))

	if likelihood_0 > likelihood_1:
		result.append(0)
	else:
		result.append(1)


# print(len(result))
print("Testing samples accuracy: ")
print(((result[:1135].count(1) + result[1135:].count(0)) / 2115) * 100)

# print(train_mat_n.shape)
result = []
for i in range(0,12665):

	x = ((train_mat_n[:,i].reshape(2,1) - u_mat_1).T @ np.linalg.inv(sigma_1)) @ (train_mat_n[:,i].reshape(2,1) - u_mat_1)

	likelihood_1 = 1/math.sqrt(4 * (math.pi**2) * sigma_det_1) \
	* math.exp(-0.5 * x)
 

	likelihood_0 = 1/math.sqrt(4 * (math.pi**2) * sigma_det_0) \
	*	math.exp(-0.5 * (train_mat_n[:,i].reshape(2,1) - u_mat_0).T @ np.linalg.inv(sigma_0) @ (train_mat_n[:,i].reshape(2,1) - u_mat_0))

	if likelihood_0 > likelihood_1:
		result.append(0)
	else:
		result.append(1)
print("Training samples accuracy: ")
print(((result[:6742].count(1) + result[6742:].count(0)) / 12665) * 100)




