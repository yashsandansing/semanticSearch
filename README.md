# semanticSearch
A semantic search that fetches clothes relevant to user's query. 

Uses a Bert-encoder to retrieve N most similar items.

Preprocessing includes: Removing punctuations, stopwords and converting to lowercase

The top-N results are set to 5 by default.

Example execution: `similarityMain.py --input "A red floral dress" --N 5`

URL for execution: https://us-central1-profound-photon-387506.cloudfunctions.net/semantic-similarity

Just enter your query after the URL in the format ?input=query

E.g usage: https://us-central1-profound-photon-387506.cloudfunctions.net/semantic-similarity?input=sundress
