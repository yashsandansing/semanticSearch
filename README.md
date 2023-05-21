# semanticSearch
A semantic search that fetches clothes relevant to user's query. 

Uses a Bert-encoder to retrieve N most similar items.

Preprocessing includes: Removing punctuations, stopwords and converting to lowercase

The top-N results are set to 5 by default.

Example execution: `similarityMain.py --input "A red floral dress" --N 5`
