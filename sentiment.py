import os
import streamlit as st
from langchain.chains import create_tagging_chain
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from nltk.sentiment import SentimentIntensityAnalyzer
from langchain_community.document_loaders import WebBaseLoader
import time
from dotenv import load_dotenv
import re
import nltk
nltk.download('vader_lexicon')


load_dotenv()

all_reviews=[]


schema = {
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "stars": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "Describes the number of stars given by a reviewer on Amazon",
        },
    },
    "required": ["sentiment", "stars"],
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
chain = create_tagging_chain(schema, llm)

def remove_markdown(text):
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'([*_]{1,2})(.*?)\1', r'\2', text)
    text = re.sub(r'#+\s+(.*?)\n', r'\1\n', text)
    text = re.sub(r'(\*|-|\+)\s+(.*?)\n', r'\2\n', text)
    text = re.sub(r'(\d+\.\s+)(.*?)\n', r'\2\n', text)
    text = re.sub(r'\*{3,}|-{3,}|_{3,}', '', text)
    return text

def extract_review_url(amazon_url):
    try:
        domain = amazon_url.split('/')[2]
        if domain == 'www.amazon.com' or domain == 'www.amazon.in':
            asin_index = amazon_url.find('/dp/') + 4
            asin = amazon_url[asin_index:asin_index + 10]
            review_url = f"https://{domain}/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
            print(review_url)
            return review_url
        else:
            print("Unsupported Amazon domain. Please provide a URL for Amazon.com or Amazon.in.")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def scrape_reviews(url):
    all_reviews = []
    try:
        loader = WebBaseLoader(url)
        soup = loader.scrape()
        reviews = soup.find_all("span", class_="a-size-base review-text review-text-content")
        for review in reviews:
            all_reviews.append(review.text.strip())
    
        next_button = soup.find('li', {'class': 'a-last'})
        if next_button and next_button.find('a'):
            next_page_url = 'https://www.amazon.com' + next_button.find('a')['href']
            time.sleep(3)
            all_reviews += scrape_reviews(next_page_url)
    except Exception as e:
        print(f"Error: {e}")
    
    return all_reviews


def analyze_review(all_reviews):
    sia = SentimentIntensityAnalyzer()
    
    compound_scores = []
    for review in all_reviews:
        sentiment_score = sia.polarity_scores(review)
        compound_scores.append(sentiment_score['compound'])
    
    average_sentiment = sum(compound_scores) / len(compound_scores)
    average_percent = round(average_sentiment * 100)
    positive_reviews = sum(score > 0 for score in compound_scores)
    negative_reviews = sum(score < 0 for score in compound_scores)
    
    return average_sentiment, positive_reviews, negative_reviews, average_percent

def analyze_reviews(all_reviews):
    sia = SentimentIntensityAnalyzer()
    
    reviews_with_scores = []
    for review in all_reviews:
        sentiment_score = sia.polarity_scores(review)
        compound_score = float(sentiment_score['compound'])  # Convert score to float
        reviews_with_scores.append((review, compound_score))
    
    return reviews_with_scores


def analyze_sentiment(review):
    try:
        response = chain.invoke(review)
        
        sentiment = response['text']['sentiment']

        # Assign stars based on sentiment
        if sentiment == 'positive':
            stars = 5
        elif sentiment == 'neutral':
            stars = 3
        else:
            stars = 1

        return sentiment, stars

    except Exception as e:
        st.error("An error occurred during sentiment analysis: " + str(e))
        return None, None
def main():
    st.set_page_config(layout='wide')
    st.title("Amazon Product Review Analyzer")
    
    tabs = ["Amazon URL", "Review"]
    choice = st.sidebar.selectbox("Choose:", tabs)

    if choice == "Amazon URL":
        st.subheader("Analyze Amazon Product URL")

        product_url = st.text_input("Enter the URL of the product on Amazon:")
        
        if st.button("Analyze URL"):
            if product_url:
                new_url = extract_review_url(product_url)
                all_reviews = scrape_reviews(new_url)
                # st.write(all_reviews)

                if all_reviews:
                    average_sentiment, positive_reviews, negative_reviews, average_percent = analyze_review(all_reviews)
                    
                    st.subheader("Sentiment Analysis Results:")
                    st.write(f"Average Sentiment Percent: {average_percent}%")
                    st.write(f"Number of Positive Reviews: {positive_reviews}")
                    st.write(f"Number of Negative Reviews: {negative_reviews}")
                    
                    if average_sentiment > 0:
                        st.success("Positive Sentiment")
                    elif average_sentiment < 0:
                        st.error("Negative Sentiment")
                    else:
                        st.info("Neutral Sentiment")
                    
                    
                    review_scores = analyze_reviews(all_reviews)
                    review_scores = sorted(review_scores, key=lambda x: x[1], reverse=True)
                    
                    st.session_state['positive_reviews'] = review_scores[:5]
                    st.session_state['negative_reviews'] = review_scores[-5:]

                    if 'positive_reviews' in st.session_state:
                        top_positive_reviews = st.session_state.positive_reviews
                    if 'negative_reviews' in st.session_state:
                        top_negative_reviews = st.session_state.negative_reviews
                    st.subheader("Top Reviews:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Top Positive Reviews:")
                        for review, score in top_positive_reviews:
                            formatted_review = remove_markdown(review)
                            st.write(f"<p style='text-align: justify;'>Review: {formatted_review}</p>", unsafe_allow_html=True)
                            percent=score*100
                            st.write(f"Sentiment Percent:{int(percent)}%")


                    with col2:
                        st.subheader("Top Negative Reviews:")
                        for review, score in top_negative_reviews:
                            formatted_review = remove_markdown(review)
                            st.write(f"<p style='text-align: justify;'>Review: {formatted_review}</p>", unsafe_allow_html=True)
                            percent=score*100

                            st.write(f"Sentiment Percent:{int(percent)}%")
                            


                else:
                    st.warning("No reviews found for the provided URL. Please make sure it is a valid Amazon product page.")
            else:
                st.warning("Please enter a valid Amazon product URL.")
    
    elif choice == "Review":
        st.subheader("Analyze Review")

        review_text = st.text_area("Please enter your review:", height=100)

        if st.button("Analyze Review"):
            if review_text:
                sentiment, stars = analyze_sentiment(review_text)
                response_text = f"Generated response based on sentiment: Sentiment: {sentiment}, Stars: {stars}"
                st.write(response_text)
            else:
                st.warning("Please enter a review.")

if __name__ == "__main__":
    main() 

