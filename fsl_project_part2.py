import scipy.io
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
import time

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


n = 6742
train_mat_1 = np.zeros((2,n))
for i in range(0,n):
	train_mat_n[0][i] = avg_pixels_sum[i]
	train_mat_n[1][i] = avg_var_sum[i]
	train_mat_1[0][i] = avg_pixels_sum[i]
	train_mat_1[1][i] = avg_var_sum[i]

###############################
# Training set 0
###############################
n = 5923
Train_data_0_reshaped=Train_data_0.reshape(784,n)
pixels_sum=[ sum(row[i] for row in Train_data_0_reshaped) for i in range(len(Train_data_0_reshaped[0])) ]
avg_pixels_sum= [x /784 for x in pixels_sum]
# print(avg_pixels_sum[0])

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

################################
# Creating test matrix
###############################
#### 1
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

# print(train_mat_n.shape)
# print(test_mat.shape)

################################
# Classifier
################################

k_list = [1, 3, 5, 7, 9, 11, 13, 15]
time_list = []
accuracy_list = []
print("Classifier is running...")
print("K - Accuracy")
for k in k_list:
	output = []
	start_time = time.time()
	for dte in test_mat.T:
		distances = []
		i = 0
		for dtr in train_mat_n.T:
			dist = math.sqrt(((dte[0] - dtr[0])**2) + ((dte[1] - dtr[1])**2));
			distances.append((dist, i))
			i+=1
		distances.sort(key=lambda tup: tup[0])

		classvotes = []
		for j in range(k):
			classvotes.append(distances[j][1])

		classvotes.sort()

		if classvotes[(int)(k/2)] <= 6741:
			output.append(1)
		else:
			output.append(0)

	count = 0
	for i in range(len(output)):
		if i < 1135:
			if output[i] == 1:
				count += 1
		else:
			if output[i] == 0:
				count += 1

	accuracy = count/len(output) * 100
	accuracy_list.append(accuracy)
	time_list.append((time.time() - start_time)/60)
	print(k, accuracy)

print(time_list)
plt.scatter(k_list, accuracy_list)
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()

plt.scatter(k_list, time_list)
plt.xlabel('K Value')
plt.ylabel('Time Taken (minutes)')
plt.show()