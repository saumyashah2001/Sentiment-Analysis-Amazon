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
from langchain_community.callbacks import get_openai_callback
import anthropic
import csv
import pandas as pd


api_key = os.environ.get("ANTHROPIC_API_KEY")


client = anthropic.Anthropic(api_key=api_key)









nltk.download('vader_lexicon')


load_dotenv()

all_reviews=[]

schema = {
    "properties": {
        "sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]},
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
    global all_reviews
    try:
        # st.write(url)
        loader = WebBaseLoader(url)
        soup = loader.scrape()
        reviews = soup.find_all("span", class_="a-size-base review-text review-text-content")
        # st.write(reviews)
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
        with get_openai_callback() as cb:
            response = chain.invoke(review)
            sentiment = response['text']['sentiment']

            # Assign stars based on sentiment
            if sentiment == 'positive':
                stars = 5
            elif sentiment == 'neutral':
                stars = 3
            else:
                stars = 1

            response_text = f"Generated response based on sentiment: Sentiment: {sentiment}, Stars: {stars}"
            st.write(response_text)
            st.subheader("Cost Analysis")

            if cb and hasattr(cb, "total_cost"):
                st.session_state['successful_requests'] = st.session_state.get('successful_requests', 0) + 1
                st.session_state['total_cost'] = st.session_state.get('total_cost', 0) + cb.total_cost
                st.write("Tokens Used:", cb.prompt_tokens + cb.completion_tokens)
                st.write("Prompt Tokens:", cb.prompt_tokens)
                st.write("Completion Tokens:", cb.completion_tokens)
                st.write("Successful Requests:", st.session_state.successful_requests)
                st.write("Total Cost (USD):", f"${st.session_state.total_cost}")

            return sentiment, stars

    except Exception as e:
        st.error("An error occurred during sentiment analysis: " + str(e))
        return None, None

       



def analyze_sentiment_and_tone(review):
    try:
        
        with get_openai_callback() as cb:
            response = chain.invoke(review)
            sentiment = response['text']['sentiment']
            if sentiment == "Positive":
                        st.success("Positive Sentiment")
            elif sentiment == "Negative":
                            st.error("Negative Sentiment")
            else:
                            st.info("Neutral Sentiment")


            
            st.subheader("Cost Analysis")

            if cb and hasattr(cb, "total_cost"):
                st.session_state['successful_requests'] = st.session_state.get('successful_requests', 0) + 1
                st.session_state['total_cost'] = st.session_state.get('total_cost', 0) + cb.total_cost
                st.write("Tokens Used:", cb.prompt_tokens + cb.completion_tokens)
                st.write("Total Cost (USD):", f"${st.session_state.total_cost}")
               
            return sentiment

    except Exception as e:
        st.error("An error occurred during sentiment analysis: " + str(e))
        return None, None, None

  
def analyze_sentiment_claude(review):
    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            system="Your task is to analyze the provided review and identify sentiment. The sentiment should be classified as Positive, Negative, or Neutral.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": review
                        }
                    ]
                }
            ]
        )
        # Calculate token usage based on the generated text
        generated_text = message.content[0].text
        tokens_used = len(generated_text.split()) if generated_text else 0
        
        # Calculate cost
        price_per_1000_tokens = 0.015  # Price per 1000 input tokens
        cost = (tokens_used / 1000) * price_per_1000_tokens
        
        # Extract sentiment from the content
        sentiment = message.content[0].text.split('.')[0].split()[-1]
        
        if sentiment == "Positive":
                        st.success("Positive Sentiment")
        elif sentiment == "Negative":
                        st.error("Negative Sentiment")
        else:
                        st.info("Neutral Sentiment")

        
        st.subheader("Cost Analysis")
                  



        st.write("Tokens Used:", tokens_used)
        st.write("Total Cost (USD):", f"${cost}")
        time.sleep(5)
        
        return sentiment
    except Exception as e:
        print("An error occurred:", e)
        return None

def analyze_openai_sentiment(reviews):
    for i, review in enumerate(reviews[:20]):
        st.write(f"**Review {i+1}:{review}**")
        sentiment= analyze_sentiment_and_tone(review)
        
        if sentiment:
             pass
            
        
            
        else:
            st.write("Failed to analyze sentiment.")

def analyze_claude_sentiment(reviews):
    for i, review in enumerate(reviews[:20]):
        st.write(f"**Review {i+1}:{review}**")
        sentiment = analyze_sentiment_claude(review)
        time.sleep(5)

        if sentiment:
            pass
           
        else:
            st.write("Failed to analyze sentiment.")





def main():
    st.set_page_config(layout='wide')
    st.title("Amazon Product Review Analyzer")
    
    tabs = ["Amazon URL","Review from Chatgpt","Review from Claude","Comparision in sentiment"]
    choice = st.sidebar.selectbox("Choose:", tabs)

    if choice == "Amazon URL":
        st.subheader("Analyze Amazon Product URL")

        product_url = st.text_input("Enter the URL of the product on Amazon:")
        
        if st.button("Analyze URL"):
            if product_url:
                new_url = extract_review_url(product_url)
                all_reviews = scrape_reviews(new_url)
                

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
    
    elif choice == "Review from Chatgpt":
        st.subheader("Analyze Review")

        review_text = st.text_area("Please enter your review:")

        if st.button("Analyze Review"):
            if review_text:
                sentiment, stars= analyze_sentiment(review_text)
                
            else:
                st.warning("Please enter a review.")
    elif choice == "Review from Claude":
        st.subheader("Analyze Review")

        review_text = st.text_area("Please enter your review:")

        if st.button("Analyze Review"):
            if review_text:
                sentiment= analyze_sentiment_claude(review_text)
                
                
            else:
                st.warning("Please enter a review.")
    elif choice == "Comparision in sentiment":
        st.subheader("Sentiment Analysis Comparison")
        amazon_url = st.text_input('Enter Amazon URL:')
        if st.button("Extract Reviews"):
            if amazon_url:
                review_url = extract_review_url(amazon_url)
                if review_url:
                    reviews = scrape_reviews(review_url)
                    if reviews:
                        df = pd.DataFrame({"review": reviews})
                        file_path = "extracted_reviews.csv"
                        df.to_csv(file_path, index=False)
                        st.success(f"Reviews extracted and stored in {file_path}")
                    else:
                        st.error("Failed to extract reviews from the provided URL.")
                else:
                    st.error("Unsupported Amazon domain or invalid URL. Please provide a URL for Amazon.com or Amazon.in.")
            else:
                st.warning("Please enter an Amazon URL.")

        if st.button("Perform Sentiment Analysis"):
            try:
                df = pd.read_csv("extracted_reviews.csv")
                if 'review' in df.columns:
                    reviews = df['review'].tolist()


                    st.write("## Sentiment Analysis")
                    
                    # Split the screen layout into two columns
                    col1, col2 = st.columns(2)
                    
                    # Perform sentiment analysis from OpenAI
                    with col1:
                        st.write("## OpenAI")
                        analyze_openai_sentiment(reviews)
                    
                    # Perform sentiment analysis from Claude AI
                    with col2:
                        st.write("## Claude AI")
                        analyze_claude_sentiment(reviews)
                        time.sleep(3)
                else:
                    st.error("CSV file doesn't contain 'review' column.")
            except FileNotFoundError:
                st.error("Please extract reviews first.")

            
        
       

              

if __name__ == "__main__":
    main() 