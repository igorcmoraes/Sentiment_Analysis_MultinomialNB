#Importando todos os módulos a serem utilizados no programa
import pandas as pd
import tweepy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from unicodedata import normalize
import matplotlib.pyplot as plot
import os
from optparse import OptionParser

if __name__ == "__main__":
	#BLOCO DE TREINO
	dataset = pd.read_csv('politicabase.csv', encoding='ISO-8859-1') #Lendo a base de dados
	dataset.count() #Contando o numero de linhas

	dataset2 = pd.read_csv('sentimentobase.csv', encoding='ISO-8859-1') #Lendo a base de dados


	tweets = dataset['texto'].values #Armazenando todos os tweets
	classes = dataset['classificacao'].values #Armazenando todas as classificações
	#Configurando o vetorizador para separar palavra por palavra
	vectorizer = CountVectorizer(analyzer="word")
	#Aplicando o vetorizador
	freq_tweets = vectorizer.fit_transform(tweets)
	#Definindo o método 
	modelo = MultinomialNB()
	#Aplicando o método, com os tweets vetorizados e as classificações 
	modelo.fit(freq_tweets,classes)

	# Pegando as opcoes pelo argv.
	parser = OptionParser()
	parser.add_option("-u", "--user", dest = "user")
	parser.add_option("-v", "--video", dest = "video")

	(options, args) = parser.parse_args()

	canal = options.user
	id_video = options.video

	if canal == "":
		print("Falta nome de usuario.\n\tpython2.7 " + " ".join(sys.argv) + " -u USUARIO")
		exit()
	
	if id_video == "":
		print("Falta id do video.\n\tpython2.7 " + " ".join(sys.argv) + " -v ID_DO_VIDEO")
		exit()

	arquivo_diretorio = "canais/" + canal + "/" + id_video + "/"

	#BLOCO DE LEITURA DO TESTE 
	arquivo = open (canal + id_video + '.txt', 'r', encoding="utf8")
	lower = arquivo.read()
	lower = lower.replace(' \n', '\n')
	lower = lower.split('\n')

	#BLOCO DE CONVERSÃO EM LOWER-CASE
	lower = [x.lower() for x in lower]

	#BLOCO DE REMOVER PONTUAÇÃO
	punct = set(string.punctuation)
	ponto = lower
	ponto = [''.join(c for c in s if c not in string.punctuation) for s in ponto]
	ponto = [s for s in ponto if s]

	#BLOCO DE REMOVER ACENTUAÇÃO 
	def remover_acentos(palavras):
		for palavra in palavras:
			result = [normalize('NFKD', palavra).encode('ASCII', 'ignore').decode('ASCII')]
			retorno = ' '.join(result)
			final.append(retorno)
	final = []
	remover_acentos(ponto)   

	#BLOCO DE REMOVER STOPWORDS
	arquivo = open ('stopwords.txt', 'r', encoding="utf8")
	stopwords = arquivo.read()
	stopwords = stopwords.replace(' \n', '\n')
	stopwords = stopwords.split('\n')
	testes = []
	palavras = final
	palavras_para_remover = stopwords
	for palavra in palavras:
		lista_frase = palavra.split()
		result = [palavra for palavra in lista_frase if palavra.lower() not in palavras_para_remover]
		retorno = ' '.join(result)
		testes.append(retorno)
		
	#BLOCO DE APLICAÇÃO DO MODELO NA BASE DE TESTE
	freq_testes = vectorizer.transform(testes)
	modelo.predict(freq_testes)



	#CONTANDO AS OCORRÊNCIAS DE CADA CLASSE
	vetor = modelo.predict(freq_testes)
	direita = 0
	esquerda = 0
	neutro = 0
	for i in range(1, len(vetor)-1):
		if vetor[i] == 'DIR':
			direita +=1
		elif vetor[i] == 'ESQ':
			esquerda +=1
		else:
			neutro +=1

	#PLOTANDO O GRÁFICO DA POLÍTICA
	x_list = [direita, esquerda, neutro]
	labels_list = ['Direita', 'Esquerda', 'Neutro']
	plt.axis('equal')
	plt.pie(x_list, labels=labels_list, colors=['blue','green','silver'], autopct='%1.2f%%')
	plt.title('Análise Política')
	plt.savefig(arquivo_diretorio + 'politica.jpg')
	plt.close()

	#TERMINOU A ANÁLISE DE SENTIMENTO! ABAIXO A ANÁLISE DE WORDCLOUD. 
	arquivo = open('wordcloud.txt','w', encoding='utf8')
	arquivo.writelines(["%s\n" % item  for item in testes])
	arquivo.close()
	text = open('wordcloud.txt','r', encoding='utf8').read()
	wordcloud = WordCloud(max_font_size=100,width = 1520, height = 535).generate(text)
	plt.figure(figsize=(16,9))
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.savefig(arquivo_diretorio + 'wordcloud.jpg')
	plt.close()

	dataset = dataset2
	dataset.count() #Contando o numero de linhas
	tweets = dataset['texto'].values #Armazenando todos os tweets
	classes = dataset['classificacao'].values #Armazenando todas as classificações
	#Configurando o vetorizador para separar palavra por palavra
	vectorizer = CountVectorizer(analyzer="word")
	#Aplicando o vetorizador
	freq_tweets = vectorizer.fit_transform(tweets)
	#Definindo o método 
	modelo = MultinomialNB()
	#Aplicando o método, com os tweets vetorizados e as classificações 
	modelo.fit(freq_tweets,classes)

	#APLICAÇÃO DO MODELO
	freq_testes = vectorizer.transform(testes)
	modelo.predict(freq_testes)

	#Contando as ocorrências de cada classe
	import matplotlib.pyplot as pyplot

	vetor = modelo.predict(freq_testes)
	negativo = 0
	positivo = 0
	neutro = 0
	for i in range(1, len(vetor)-1):
		if vetor[i] == 'NEG':
			negativo +=1
		elif vetor[i] == 'POS':
			positivo +=1
		else:
			neutro +=1

	#Plotando os gráficos
	x_list = [negativo, positivo, neutro]
	labels_list = ['Negativo', 'Positivo', 'Neutro']
	plt.axis('equal')
	plt.pie(x_list, labels=labels_list, colors=['red','blue','silver'], autopct='%1.2f%%')
	plt.title('Análise de Sentimentos')
	plt.savefig(arquivo_diretorio + 'sentimento.jpg')
	plt.close()
