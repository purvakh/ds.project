import numpy as np
import math

class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#training logic here
		#input is an array of features and labels
		self.merged_data = np.concatenate((X,y[:,None]),axis=1)
		None

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		outputlist = []
		for testrow in X:
			currlist=[]
			for trainrow in self.merged_data:
				currlist.append((self.distance(testrow,trainrow[:-1]),trainrow[-1]))
			currlist.sort(key=lambda x:x[0])
			curr = currlist[:self.k]
			outputs = [x[1] for x in curr]
			
			z = max(set(outputs), key= outputs.count)
			outputlist.append(z)
		return np.array(outputlist)

class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		merged_data = np.concatenate((categorical_data,y[:,None]),axis=1)
		
		first,eninfo = self.calculateInfoGain(merged_data)
		self.head = Node(first)
		self.split(first,merged_data,self.head)
	
	def split(self,split_atr,df,node):
		atr = np.unique(df[:,:-1][:,split_atr])
		labelcount = df[:,-1]
		node.label = max(set(labelcount), key = list(labelcount).count)
		if len(np.unique(df[:,-1]))==1:
			node.isterminal=True
			node.label = int(np.unique(df[:,-1]))
			return
		
		if len(atr) == 1:
			node.isterminal=True
			labelcount = df[:,-1]
			node.label = max(set(labelcount), key = list(labelcount).count)
			return 
		else:
			splitdf=[]
			for i in atr:
				splitdf.append((df[df[:,split_atr]==i],i))
			for dfs in splitdf:
				val,eninfo = self.calculateInfoGain(dfs[0])
				#print(val,eninfo,dfs[1],depth,atr)
				new = Node(val)
				node.childlist.append((new,dfs[1]))
				self.split(val,dfs[0],new)	
	
	def calculateInfoGain(self,merged_data):
		bins_value = np.unique(merged_data[:,:-1])
		p_entropy = self.entropy(merged_data)
		info_gain=[]
		r_idx,c_idx = merged_data.shape
		for i in range(c_idx-1):
			total = 0
			en_list=[]
			for j in bins_value:
				no = len(merged_data[merged_data[:,i]==j])
				eno = self.entropy(merged_data[merged_data[:,i]==j])
				total+=no
				en_list.append((no,eno))
			gain = p_entropy
			for index,value in enumerate(en_list):
				gain-= value[0]*value[1]/total
			info_gain.append((i,gain))
		info_gain.sort(key = lambda x:x[1], reverse=True)	
		return info_gain[0][0],info_gain[0][1]
	
	def entropy(self,docset):
		no1 = len(docset[docset[:,-1]==0])
		no2 = len(docset[docset[:,-1]==1])
		entropy = 0
		if 0 in [no1,no2]:
			return 0
		for i in [no1,no2]:
			entropy+=-1*(i/(no1+no2)*math.log2(i/(no1+no2)))
		return entropy
	
	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		outputlist = []
		for row in categorical_data:
			outputlist.append(self.treeflow(row))
		return np.array(outputlist)
	
	def treeflow(self,row):
		node = self.head
		while not node.isterminal:
			atr = node.split
			getval = row[atr]
			childlist = [child[1] for child in node.childlist]
			if getval not in childlist:
				return node.label
			else:
				for child in node.childlist:
					if child[1]==getval:
						node=child[0]
			
		return node.label
		
class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b


	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		w=self.w
		b=self.b
		r_idx,c_idx = X.shape
		step = int(steps/r_idx)
		for _ in range(step):
			for index,row in enumerate(X):
				z=row.dot(w)+b
				if int(np.sign(z))==-1:
					d=0
				else:
					d=1
				err=y[index]-d
				w+=self.lr*err*row
				b+=self.lr*err
				
		self.w = w
		self.b = b
		
	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		output=[]
		for row in X:
			z=row.dot(self.w)+self.b
			if int(np.sign(z))==-1:
				d=0
			else:
				d=1
			output.append(int(d))
			
		return np.array(output)

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y): 
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)
			
			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi) 
		#	print(loss)
			
			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)


			
	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		self.input = input
		value = input.dot(self.w)+self.b
		return value

	def backward(self, gradients):
		#Write backward pass here
		deltaw = self.input.T.dot(gradients)
		value_ = gradients.dot(self.w.T)
		self.w-=self.lr*deltaw
		self.b-=self.lr*np.sum(deltaw,axis=0)
		return value_
		
class Node:

	def __init__(self, split_atr):
		self.split = split_atr
		self.childlist=[]
		self.label = None
		self.isterminal = False
	
class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		self.sigmoid = 1 / (1 + np.exp(-input))
		self.input = input
		return self.sigmoid

	def backward(self, gradients):
		#Write backward pass here
		backward = gradients*(1-self.sigmoid)*self.sigmoid
		return backward