from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
import numpy
# fix random seed for reproducibility
numpy.random.seed(6)

dataset = numpy.loadtxt("dataset_treino.csv", delimiter=",")
dataset_teste = numpy.loadtxt("dataset_teste.csv", delimiter=",")
def NeuralNetwork(dataset,epoc,percent):
	# p = tamanho de x% do dataset e l = tamanho do dataset
	p = int(len(dataset)*percent)
	l = len(dataset)
	#x_treino = entradas de treino da rede e y_treino = saida esperada da rede
	x_train = dataset[0:p,1:6]
	y_train = dataset[0:p,6]
	#x_teste = entradas pra teste da rede e y_teste = saida esperada da rede
	x_test = dataset[p:l,1:6]
	y_test = dataset[p:l,6]
	#cria o modelo da rede com 2 camadas ocultas de 9 e 6 neuronios e a de saida com 1
	model = Sequential()
	model.add(Dense(7, input_dim = 5, use_bias = True))
	model.add(LeakyReLU(0.02))
	model.add(Dense(7,use_bias = True))
	model.add(LeakyReLU(0.02))
	model.add(Dense(1, activation='hard_sigmoid',use_bias = True))
	#compila o modelo e da uma nota pra rede
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=epoc)
	#volta o loss e o acc
	print("teste:")
	scores = model.evaluate(x_test, y_test)
	print("\nteste %s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
	print("\nteste %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	#prediction
	predictions = model.predict(dataset_teste[:,1:6])
	rounded = [round(x[0]) for x in predictions]
	float2string(rounded)
#printa o resultado convertendo os valores float pra uma resposta em defensor ou atacante
def float2string(rounded):
	pred_final=[]
	for x in rounded:
		if x == 1.0: pred_final.append('atacante')
		else : pred_final.append('defensor')
	print(pred_final)
#chama a função Neural Network
NeuralNetwork(dataset,150,0.8)
