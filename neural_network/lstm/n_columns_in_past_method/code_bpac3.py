import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import root_mean_squared_error
import plotly.express as px
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
start = datetime(2021, 1, 1)
end = datetime(2023, 12, 30)

bpac3 = yf.Ticker("BPAC3.SA")

max_days = 90
df_bpac3_original = bpac3.history(start=start-relativedelta(days=max_days*2),end=end)

df_bpac3 = df_bpac3_original
# Removendo as colunas que não será necessárias
df_bpac3 = df_bpac3.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])


# Responsável por fazer a divisão de elementos de eixo x(High e Low) e y(Close)
def split_sequence(sequence, n_steps, X_in, y_in):
	X, y = X_in, y_in
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return X, y

legs = 21
qtd_goal = 1 # Pois quer predizer apenas 1, o close
matrix = {"janela":[], "rmse": []}
matrix_big = {}
for epoch in range(500, 700, 100):
  df_bpac3 = df_bpac3_original

  df_bpac3 = df_bpac3.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
  hs = open("hst.txt","a")
  hs.write(f"\n{matrix_big}")
  print(matrix_big)
  hs.close()
  for janela in range(15, max_days+15, 15):
      qtd_features = janela


      # Adiciona novas colunas com os valores anteriores
      for column_id in range(1, janela+1):
          df_bpac3[[f'days_before_{column_id}']] = df_bpac3_original[['Close']].shift(column_id)
      # Listagem do nome das novas colunas que foram inseridas
      columns_inserted = list(df_bpac3.columns)
      columns_inserted.pop(0) #Remove Close column


      # Volta para o range que estamos estudando 2021-2023
      df_bpac3 = df_bpac3.loc[df_bpac3.index > '2021-08-01']
      total_rows = len(df_bpac3)


      # Divisão de elementos de treinamento e teste
      df_train = df_bpac3.iloc[:total_rows - legs]
      df_test = df_bpac3.iloc[total_rows - legs:]

      # Listagem ordenada de colunas que será usadas no modelo
      columns_to_model = columns_inserted + ['Close']

      training_set_feature = df_train[columns_to_model].values
      testing_set_feature = df_test[columns_to_model].values

      X, y = list(), list()

      # Pega os valores de High e Low e coloca em X e Close em y para todos os elementos de treinamento
      for index in range(len(training_set_feature)):
          X, y = split_sequence(training_set_feature[index], qtd_features, X, y)
      X, y = np.array(X), np.array(y)

      #Cria o modelo
      rnn = Sequential()

      rnn.add(LSTM(units=1, return_sequences=True, input_shape=(qtd_features, qtd_goal)))
      rnn.add(Dropout(0.5))

      rnn.add(LSTM(units=50, return_sequences=True))
      rnn.add(Dropout(0.5))

      rnn.add(LSTM(units=50, return_sequences=True))
      rnn.add(Dropout(0.5))

      rnn.add(LSTM(units=50, return_sequences=True))
      rnn.add(Dropout(0.5))

      rnn.add(LSTM(units=50))
      rnn.add(Dropout(0.5))

      rnn.add(Dense(units=1))
      rnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

      rnn.fit(X, y, epochs = epoch, batch_size = 32)

      # Testa o elemento com os valores de High e Low para receber um valor de Close

      y_real_test = []
      y_predicted_test = []
      for value in testing_set_feature:

          # Pegando os valores de cada coluna que não é a target(Close)
          days_before = value[0:-1]
          close_real = value[-1]
          x_input = np.array([days_before])

          x_input = x_input.reshape((qtd_goal, qtd_features))
          yhat = rnn.predict(x_input, verbose=0)

          y_real_test.append(close_real)
          y_predicted_test.append(yhat[0][0])

      rmse = root_mean_squared_error(y_real_test, y_predicted_test)
      if epoch not in matrix_big:
        matrix_big[epoch] = {"janela":[], "rmse": []}

      matrix_big[epoch]['janela'].append(janela)
      matrix_big[epoch]['rmse'].append(rmse)

print(matrix_big)