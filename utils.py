from app import db,User,Papers

def mcda(df):
	import numpy as np
	feature = 4
	m = len(df)
	utility = np.zeros(m)
	weight = np.zeros(feature)
	maxm = [1000,1000,1000,1000]
	total = 0
	for i in range(feature):
		for j in range(m):
			r = df.values[j,5+i]
			weight[i] += r * np.log(r/maxm[i] + 1.1) /maxm[i]
		weight[i] += 1
		total += weight[i]
	for i in range(feature):
		weight[i] = weight[i]/total
	for i in range(m):
		for j in range(feature):
			utility[i] += weight[j]*df.values[i,5+j]/maxm[j]
	tmp = np.argsort(utility)
	res = []
	for i in range(len(df)):
		res.append(tmp[-i])
	return res

def rankaggr_brute(df):
	import numpy as np
	feature = 4
	m = len(df)
	utility = np.zeros(m)
	weight = np.zeros(feature)
	maxm = [1000,1000,1000,1000]
	total = 0
	for i in range(feature):
		for j in range(m):
			r = df.values[j,5+i]
			weight[i] += r * np.log(r/maxm[i] + 1.1) /maxm[i]
		weight[i] += 1
		total += weight[i]
	for i in range(feature):
		weight[i] = weight[i]/total
	for i in range(m):
		for j in range(feature):
			utility[i] += weight[j]*df.values[i,5+j]/maxm[j]
	tmp = np.argsort(utility)
	res = []
	for i in range(len(df)):
		res.append(tmp[-i])
	return res



def lstm(papers):
	from tensorflow import keras
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import LSTM
	from sklearn.preprocessing import LabelEncoder
	import numpy as np
	inputs = 10
	Xin = []
	Yin = []
	for i in range(len(papers)-inputs):
	    Xi = []
	    for j in range(inputs):
	        X = [papers[i+j][3], papers[i+j][6], papers[i+j][5], papers[i+j][4],0,0,0,0,0]
	        #X = [papers[i+j].KDM, papers[i+j].SCA, papers[i+j].SQM, papers[i+j].CAOT,0,0,0,0,0] ##############
	        n = papers[i+j][7] ###########################
	        try:
	            X[3+n] = 1
	        except:
	            X[3+int.from_bytes(n,"little" )] = 1
	        Xi.append(X)
	    Xin.append(Xi)
	for i in range(inputs,len(papers)):
	    Yi = [0,0,0,0,0]
	    try:
	        Yi[papers[i][7] - 1] = 1 ###############
	    except:
	        Yi[int.from_bytes(papers[i][7],"little" ) - 1] = 1 ####################
	    Yin.append(Yi)

	Yin = np.array(Yin)
	Xin = np.array(Xin)
	model = Sequential()
	model.add(LSTM(50,return_sequences=True, input_shape=(Xin.shape[1], Xin.shape[2])))
	model.add(LSTM(10,return_sequences=False))
	model.add(Dense(5,activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
	model.fit(Xin, Yin, epochs = 5, shuffle=False,)
	final = np.argsort((-model.predict(Xin[-1].reshape(1,inputs,Xin.shape[2]))))[0][:2] + 1
	return final[:2]

