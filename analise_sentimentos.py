#Importando os módulos necessários.
import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
import tweepy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as pyplot


# Armazenando os dados da API:
#Obs.: Os dados abaixo são privados. São identificadores individuais de cada aplicação e do desenvolvedor que a está
#utilizando. Os dados podem ser obtidos a partir do site https://developer.twitter.com
consumerKey = "INSIRA AQUI OS SEUS DADOS IDENTIFICADORES DA API DO TWITTER"
consumerSecret = "INSIRA AQUI OS SEUS DADOS IDENTIFICADORES DA API DO TWITTER"
accessToken = "INSIRA AQUI OS SEUS DADOS IDENTIFICADORES DA API DO TWITTER"
accessTokenSecret = "INSIRA AQUI OS SEUS DADOS IDENTIFICADORES DA API DO TWITTER"

# Passando os dados da API para a aplicação python:
auth = OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

#Criando o arquivo para armazenar os tweets
arquivo = open('BASEDETREINO.txt', 'a', encoding='utf8')


#Criando a função que coleta os tweets. O caractere "|" foi adicionado ao final de cada tweet, pois é necessário
#um identificador do término de cada tweet para posterior separação, já que quando o algoritmo lê as bases, coloca
#todo o conteúdo em uma única string.

class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        if hasattr(status, 'retweeted_status'):
            try:
                tweet = status.retweeted_status.extended_tweet["full_text"]
                print('{}{}'.format(tweet, ' |'))
                arquivo.write('{}{}'.format(tweet, ' |'))
            except:
                tweet = status.retweeted_status.text
                print('{}{}'.format(tweet, ' |'))
                arquivo.write('{}{}'.format(tweet, ' |'))
        else:
            try:
                tweet = status.extended_tweet["full_text"]
                print('{}{}'.format(tweet, ' |'))
                arquivo.write('{}{}'.format(tweet, ' |'))
            except AttributeError:
                tweet = status.text
                print('{}{}'.format(tweet, ' |'))
                arquivo.write('{}{}'.format(tweet, ' |'))

#Chamando a API e a função de coleta
MyStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=MyStreamListener, tweet_mode="extended")

#Definindo as palavras-chave desejadas
myStream.filter(track=['INSIRA', 'AQUI', 'AS', 'PALAVRAS-CHAVE'])

#A partir daqui a base de treino do classificador está criada. As linhas 63 a 76 referem-se à criação do modelo
#classificador.

dataset = pd.read_csv('BASEDETREINO.csv') #Lendo a base de dados
dataset.count() #Contando o numero de linhas

tweets = dataset['texto'].values #Armazenando todos os tweets
classes = dataset['classificacao'].values #Armazenando todas as classificações

#Configurando o vetorizador para separar palavra por palavra
vectorizer = CountVectorizer(analyzer="word")
#Aplicando o vetorizador
freq_tweets = vectorizer.fit_transform(tweets)
#Definindo o método
modelo = MultinomialNB() #Para essa metodologia é utilizado o Naive Bayes Multinomial. Para outros, consulte a documentação do scikit-learn.
#Aplicando o método, com os tweets vetorizados e as classificações
modelo.fit(freq_tweets,classes)

#A partir daqui o modelo classificador já está pronto. Para criar a base que deverá ser rotulada pelo classificador
#(chamamos aqui de BASEDETESTE), basta repetir os mesmos passos da criação da base de treino.

arquivo = open ('BASEDETESTE.txt', 'r', encoding='utf8') #Abrindo a base de teste
planificar = arquivo.read() #Armazenando o conteúdo do arquivo em uma string
lower = lower.replace(' \n', '\n') #Padronizando a quebra de linha do documento.
planificar = planificar.replace('\n','') #Planificando a string
testes = planificar.split('|') #Separando a string em uma lista, passando como parâmetro de separação o caractere |

freq_testes = vectorizer.transform(testes) #Aplicando o modelo

#Contando as ocorrências de cada classe
vetor = modelo.predict(freq_testes)
positivo = 0
negativo = 0
neutro = 0
for i in range(1, len(vetor)-1):
    if vetor[i] == 'Positivo':
        positivo +=1
    elif vetor[i] == 'Negativo':
        negativo +=1
    else:
        neutro +=1

#Plotando os gráficos
x_list = [positivo, negativo, neutro]
labels_list = ['Positivo', 'Negativo', 'Neutro']
pyplot.axis('equal')
pyplot.pie(x_list, labels=labels_list, colors=['Green','red','silver'], autopct='%1.2f%%')
pyplot.title('Análise de Sentimentos: Durante a votação')
pyplot.show()
