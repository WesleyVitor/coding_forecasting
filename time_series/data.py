import yfinance as yf

bpac3 = yf.Ticker("BPAC3.SA")
meli34 = yf.Ticker("MELI34.SA")
pags34 = yf.Ticker("PAGS34.SA")
bpan4 = yf.Ticker("BPAN4.SA")

df_bpac3 = bpac3.history(start="2021-01-01",end="2023-12-30")
df_meli34 = meli34.history(start="2021-01-01",end="2023-12-30")
df_pags34 = pags34.history(start="2021-01-01",end="2023-12-30")
df_bpan4 = bpan4.history(start="2021-01-01",end="2023-12-30")
