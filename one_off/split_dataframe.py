import pandas as pd

df = pd.read_csv('independent/data/twitter/comments.csv', nrows=1000)
num_chunks = 80

chunk_size = int(len(df) / 80)
chunk_start = 0

for i in range(num_chunks):
    chunk_df = df[chunk_start: chunk_start + chunk_size]
    chunk_df.to_csv('chunk_' + str(i) + '.csv', index=None)
    chunk_start += chunk_size
