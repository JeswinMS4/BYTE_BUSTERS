from bs4 import BeautifulSoup
import requests
import re
import streamlit as st
import csv
import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


@st.cache_resource
def model_init():
    model_name = "human-centered-summarization/financial-summarization-pegasus"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    return tokenizer, model


@st.cache_resource
def sent_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "Venkatesh4342/bert-base-uncased-finetuned-fin")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Venkatesh4342/bert-base-uncased-finetuned-fin")

    return tokenizer, model


@st.cache_data
def scrap():
    url = "https://finance.yahoo.com/most-active"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    paragraphs = soup.find_all("a", class_="Fw(600) C($linkColor)")

    topics = []
    for i in range(len(paragraphs)):
        topics.append(paragraphs[i].text)

    topics = topics[:2]

    def stock_news_urls(topic):
        search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(
            topic)
        r = requests.get(search_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        atags = soup.find_all('a')
        hrefs = [link['href'] for link in atags]
        return hrefs

    raw_urls = {topic: stock_news_urls(topic) for topic in topics}

    exclude = ['maps', 'policies', 'preferences', 'accounts', 'support']

    def pop_unwanted_urls(urls, exclude):
        val = []
        for url in urls:
            if 'https://' in url and not any(exclude_word in url for exclude_word in exclude):
                res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                val.append(res)
        return list(set(val))
    cleaned_urls = {topic: pop_unwanted_urls(
        raw_urls[topic], exclude) for topic in topics}

    def scrape_and_process(URLs):
        ARTICLES = []
        for url in URLs:
            r = requests.get(url)
            if r.status_code < 400:
                soup = BeautifulSoup(r.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text = [paragraph.text for paragraph in paragraphs]
                words = ' '.join(text).split(' ')
                ARTICLE = ' '.join(words)
                ARTICLES.append(ARTICLE)
        return ARTICLES

    articles = {topic: scrape_and_process(
        cleaned_urls[topic]) for topic in topics}

    def filtered_url(URLs):
        filt_url = []
        for url in URLs:
            r = requests.get(url)
            if r.status_code < 400:
                filt_url.append(url)
        return filt_url

    filt_url = {topic: filtered_url(cleaned_urls[topic]) for topic in topics}

    def semmer(topic):
        sum = []
        for i in range(len(articles[topic])):
            ARTICLE = articles[topic][i]
            max_chunk = 450
            ARTICLE = ARTICLE.replace('.', '.<eos>')
            ARTICLE = ARTICLE.replace('?', '?<eos>')
            ARTICLE = ARTICLE.replace('!', '!<eos>')
            sentences = ARTICLE.split('<eos>')
            current_chunk = 0
            chunks = []
            for sentence in sentences:
                if len(chunks) == current_chunk + 1:
                    if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                        chunks[current_chunk].extend(sentence.split(' '))
                    else:
                        current_chunk += 1
                        chunks.append(sentence.split(' '))
                else:
                    print(current_chunk)
                    chunks.append(sentence.split(' '))

            for chunk_id in range(len(chunks)):
                chunks[chunk_id] = ' '.join(chunks[chunk_id])

            summaries = []
            tokenizer, model = model_init()
            for text in chunks:
                input_ids = tokenizer.encode(
                    text, return_tensors='pt', max_length=512, truncation=True)
                output = model.generate(
                    input_ids, max_length=55, num_beams=5, early_stopping=True)
                summary = tokenizer.decode(output[0], skip_special_tokens=True)
                summaries.append(summary)

            a = ' '.join([summ for summ in summaries])
            sum.append(a)

        return sum
    tokenizer1, model1 = sent_model()

    ars = {topic: semmer(topic) for topic in topics}
    nlp = pipeline("sentiment-analysis", model=model1, tokenizer=tokenizer1)
    scores = {topic: nlp(ars[topic]) for topic in topics}

    def create_output_array(ars, scores, urls):
        output = []
        for topic in topics:
            for counter in range(len(ars[topic])):
                output_this = [
                    topic,
                    ars[topic][counter],
                    scores[topic][counter]['label'],
                    scores[topic][counter]['score'],
                    filt_url[topic][counter]
                ]
                output.append(output_this)
        return output

    final_output = create_output_array(ars, scores, cleaned_urls)

    final_output.insert(0, ['Stocks', 'Summary', 'Label', 'Confidence', 'URL'])

    with open('summaries.csv', mode='w', newline='', encoding='utf8') as f:
        csv_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(final_output)

    df1 = pd.read_csv('summaries.csv', delimiter=',')

    return df1
