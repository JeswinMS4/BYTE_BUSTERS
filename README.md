# BYTE_BUSTERS
Sentiment Analysis on Summarized text of financial data and stocks.

The problem statement that we picked here is "Stock Analysis for Beginners providing efficient time management."


Here, we initially have taken data from websites such as Yahoo finance by webscraping using Beautiful Soup. Later we performed we created a list of stocks from the website about which we could extract multiple blogposts. These multiple blogposts are later made into chunks and converted into embeddings.

These embeddings are passed through the summarizer model which is Pegasus Model finetuned on Finetuned dataset based on Bloomberg which performs when on financial data.

Later, we obtain the summerized text from the model and again create embeddings which has to be passed through Sentiment Analysis model which is a finetuned model of Bert-uncased. The final data is then created as a csv file and displayed through streamlit interface.
