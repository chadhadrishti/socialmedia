import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.agents import load_tools, Tool
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain import OpenAI, LLMChain
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from streamlit import components
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts import options as opts
from pyecharts.charts import Polar
from pyecharts.globals import ThemeType
import plotly.graph_objects as go
from PIL import Image

from streamlit import config
import re
import os

from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import FAISS
import tempfile

st.set_option("deprecation.showfileUploaderEncoding", False)


# import matplotlib.pyplot as plt
# import seaborn as sns
# import nltk
# from textblob import TextBlob
# from nltk.corpus import stopwords
# from wordcloud import WordCloud
def rose_chart() -> Polar:
    c = (
        Polar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        .add_schema(
            angleaxis_opts=opts.AngleAxisOpts(
                type_="category", data=list(counts.keys()), start_angle=0, is_clockwise=True
            ),
            radiusaxis_opts=opts.RadiusAxisOpts(min_=0),
            # polar_opts=opts.PolarOpts(),
        )
        .add(
            "",
            [list(counts.values())],
            type_="bar",
            coordinate_system="polar",
            itemstyle_opts=opts.ItemStyleOpts(color="#FF6347"),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="Nightingale's Rose Chart"))
    )
    return c


def generate_charts(df):
    for i in df['Year'].unique():
        df1 = df[df["Year"] == i]
        df1['Count'] = df1['Count'].fillna(0)
        fig = px.scatter(df1, x='Quarter', y='Label', size='Count', text='Label', hover_name='Label', size_max=60)

        # Customize the plot
        fig.update_traces(textposition='top center', showlegend=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(title='', yaxis_title=i, height=500, width=1000, xaxis_title='')

        # Display the plot in Streamlit
        st.plotly_chart(fig)


# Function to clean the text
def clean_text(text):
    # Check if the input is a string
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    text = ' '.join(tokens)

    return text


# Function to create a word cloud
def create_wordcloud(text):
    # Remove the specified words from the text
    words_to_remove = ['oreo', 'toblerone']
    for word in words_to_remove:
        text = re.sub(f'\\b{word}\\b', '', text, flags=re.IGNORECASE)

    wc = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(text)
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    return wc


# Example data
df = pd.read_csv('new_data.csv')

df.columns = ['Source', 'Comment', 'Product', 'Date']

# Convert date strings to datetime objects and filter by date
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2018-01-01']

# Group data by product, source, and quarter
df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
grouped_df = df.groupby(['Product', 'Source', 'Quarter']).size().reset_index(name='Count')

df['Clean_Comment'] = df['Comment'].apply(clean_text)
topics = pd.read_csv('topic_counts.csv')
topics["Count"] = topics["Topics"].apply(lambda x: int(x.split("(")[-1].split(")")[0]))
topics["Topics"] = topics["Topics"].apply(lambda x: x.split(" (")[0])


def create_bubble_plot(df, product):
    df_filtered = df[df['Product'] == product]

    fig = px.scatter(df_filtered, x="YearQuarter", y="Count", size="Count", text="Topics", color="Topics",
                     title=f"Topics Over Time for Product {product}", height=600, width=1000)

    fig.update_traces(textposition='top center', textfont_size=10)
    fig.update_layout(showlegend=False)

    return fig


def create_sunburst_chart():
    fig = go.Figure(go.Sunburst(
        labels=[
            "Product", "Price", "Promotion", "Place",
            "Taste", "Flavours", "Texture", "Packaging", "Size",
            "Expensive", "Affordable", "Value for Money", "Worth",
            "Offers", "Discounts", "Cross selling", "Sales/Festive ordering", "Advertisements",
            "Stock Availability", "Difficult to purchase", "Specific Flavour availability", "Seasonal unavailability"
        ],
        parents=[
            "", "", "", "",
            "Product", "Product", "Product", "Product", "Product",
            "Price", "Price", "Price", "Price",
            "Promotion", "Promotion", "Promotion", "Promotion", "Promotion",
            "Place", "Place", "Place", "Place"
        ],
        values=[
            1, 1, 1, 1,
            10, 10, 10, 10, 10,
            10, 10, 10, 10,
            10, 10, 10, 10, 10,
            10, 10, 10, 10
        ],
        textfont=dict(family="Arial", size=14),
        insidetextfont=dict(family="Arial", size=16, color="white"),
        outsidetextfont=dict(family="Arial", size=16, color="black"),
    ))

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        autosize=False,
        width=800,
        height=800,
        shapes=[
            dict(
                type="circle",
                xref="paper", yref="paper",
                x0=0.425, y0=0.425, x1=0.575, y1=0.575,
                line=dict(color="white", width=3),
                fillcolor="white"
            )
        ]
    )

    return fig


# Create a donut chart
def donut_chart() -> Pie:
    c = (
        Pie()
        .add("", [(k, v) for k, v in category_counts.items()], radius=["40%", "75%"])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Donut Chart: Number of Items"),
            legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%"),
        )
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    )
    return c


# App title and configuration
st.set_page_config(page_title="Product Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("Customer Voice Analyzer")

# Load data
# data = pd.read_csv("products.csv")
# clean_data(data)

# Sidebar
st.sidebar.title("Select a Product")
products = ['A', 'B']
product = st.sidebar.selectbox("Choose a product", products)
# Filter data based on selected product
filtered_df = grouped_df[grouped_df['Product'] == product]
filtered_raw_df = df[df['Product'] == product].head(1000)

# Get all comments for selected product
text = ' '.join(df[df['Product'] == product]['Clean_Comment'])

# Tabs
# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
#     ["Approach", "Data Exporation", "Topic Analysis", "Market Mix 4Ps", "Competitive Analysis", "Co-Pilot"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Approach", "Data Exporation", "Topic Analysis", "Market Mix 4Ps", "Competitive Analysis"])

product_data = [
    "Texture and Size",
    "Unique White Chocolate",

    "Quality and Reputation"]

price_data = [
    "Price and Availability",
    "Inflation and Economy"
]

promotion_data = [
    "Drama and Excitement",

    "Promotions and Reviews"
]

placement_data = [
    "Company and Brand",

    "International Availability"
]

product_df = pd.DataFrame(product_data, columns=["Product"])
price_df = pd.DataFrame(price_data, columns=["Price"])
promotion_df = pd.DataFrame(promotion_data, columns=["Promotion"])
placement_df = pd.DataFrame(placement_data, columns=["Placement"])

# Display content based on selected tab


font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 20px;
}
</style>
"""

st.write(font_css, unsafe_allow_html=True)

with tab1:
    st.image('approach1.png', use_column_width=True)

    st.header('Machine Learning Techniques Used:')
    st.subheader("Topic Modeling")
    st.info(
        """
        Topic modeling is a machine learning technique used to analyze and categorize large volumes of text data. It identifies recurring patterns or themes, known as 'topics,' within the text.

        For example, if a business receives numerous customer reviews, the topic modeling algorithm would identify words that commonly co-occur and form topics based on these patterns. This could reveal topics like product quality, customer service, or pricing.
        """
    )

    st.subheader("Zero Shot Learning")
    st.info(
        """
        Zero-shot learning is a cutting-edge approach in machine learning where a model can recognize or categorize new, unseen data without requiring any prior training on that specific data. In other words, the model can make predictions or classifications on new information that it has never encountered before, based on its previous learning.
        """
    )

with tab2:
    st.subheader("Data Collected For Product Analysis:")
    filtered_df = grouped_df[grouped_df['Product'] == product]
    filtered_raw_df1 = df[df['Product'] == product]
    Amazon_count = filtered_raw_df1[filtered_raw_df1['Source'] == 'Amazon'].shape[0]
    Wallmart_count = filtered_raw_df1[filtered_raw_df1['Source'] == 'Wallmart'].shape[0]
    Target_count = filtered_raw_df1[filtered_raw_df1['Source'] == 'Target'].shape[0]
    Twitter_count = filtered_raw_df1[filtered_raw_df1['Source'] == 'Twitter'].shape[0]

    col11, col22, col33, col44 = st.columns(4)

    # st.write("")
    # st.write("")
    col11.image('Amazon_icon.png', width=50)
    col11.metric("Amazon", Amazon_count)
    col22.image('wallmart.png', width=50)
    col22.metric("Wallmart", Wallmart_count)
    col33.image('target.png', width=50)
    col33.metric("Target", Target_count)
    col44.image('twitter.png', width=50)
    col44.metric("Twitter", Twitter_count)

    st.subheader("Word Cloud")
    if product == 'A':
        st.image('product_a_wordcloud.png', use_column_width=True)
    if product == 'B':
        st.image('product_b_wordcloud.png', use_column_width=True)
    # wc = create_wordcloud(text)
    # st.pyplot(plt)

    # Display raw data
    st.subheader("Raw Data")
    filtered_raw_df = filtered_raw_df.replace(['Oreos', 'oreo', 'oreos'], 'A', regex=True)
    filtered_raw_df = filtered_raw_df.replace(['toblerone', 'Toblerone'], 'B', regex=True)
    st.dataframe(filtered_raw_df.head(100))

with tab3:
    # Date selector
    # Multi-select box for quarters
    # tab_t1,tab_t2= st.tabs(["Topic Charts","Actionable Insights"])

    # st.subheader("Topic Data")
    st.subheader("Bubble Plot for Topics Over Time")

    if product == 'A':
        # product_a = pd.read_csv('product_a_topics_process.csv')
        # generate_charts(product_a)
        with open("scatter_plotsa.html", "r") as f:
            html_content_t = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

        with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
            components.v1.html(html_content_t, height=1000, scrolling=True)

    if product == 'B':
        with open("scatter_plotsb.html", "r") as f:
            html_content_b = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

        with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
            components.v1.html(html_content_b, height=1000, scrolling=True)

    st.subheader("Extracted Topics Over Quarters")
    if product == 'B':
        product_b = pd.read_csv('product_b_topics.csv')
        st.dataframe(product_b)

    if product == 'A':
        product_a = pd.read_csv('product_a_topics.csv')
        st.dataframe(product_a)

    # topic_df = pd.DataFrame(data)

    st.subheader("Topic Insights For Product")
    with st.expander("Click to see insights"):
        if product == 'B':
            st.subheader('Product B:')
            st.subheader('Seasonal trends:')
            st.markdown('''
            - Holiday seasons, particularly October to December, show a significant increase in discussions related to chocolate, such as "Holiday Chocolate," "Holidays and Gifting," and "Christmas and New Ways." This indicates a higher demand for chocolate products during these periods.
            - Conversations around "New and Limited Edition" products peak in October, which suggests that customers might be interested in trying unique flavors and products during the holiday season.
            ''')

            # Customer preferences
            st.subheader('Customer preferences:')
            st.markdown('''
            - There is a consistent interest in various aspects of chocolate, such as taste, quality, and unique flavors. This implies that maintaining high quality and offering a variety of flavors would help retain customer satisfaction.
            - Topics related to "size" and "shape" appear multiple times, indicating that customers might appreciate innovative packaging and product designs.
            ''')

            # Market trends
            # st.subheader('Market trends:')
            # st.markdown('''
            # - The dataset shows a growing interest in the brand's Swiss identity and international presence, suggesting that the company could benefit from expanding its global reach and emphasizing its Swiss heritage in marketing campaigns.
            # - There is a considerable interest in special editions, limited edition, and new flavors throughout the dataset, indicating that product innovation and variety are essential in capturing the attention of customers.
            # ''')

            # Customer engagement and support
            st.subheader('Customer engagement and support:')
            st.markdown('''
            - Topics such as "Communication and Inquiry," "Language and Expressions," and "Prayers and Support" indicate that customers value engaging with the brand and appreciate the support they receive.
            - Discussions around "Shipping and Delivery," "Price and Availability," and "Product Availability" suggest the importance of efficient distribution channels and making products accessible to customers.
            ''')
        if product == 'A':
            # Seasonal trends
            st.subheader('Product A:')
            st.subheader('Seasonal trends:')
            st.markdown('''
            - In Q4 there is a significant increase in discussions around "Christmas Treats", "New Experiences" and "Festive", suggesting a higher demand for themed or seasonal flavors during specific times of the year. 
            - The data suggests Product A is also strongly associated with themes around “Family”, “Party” and “Sharing” which implies that the business could benefit from offering products in family packs, suitable for gatherings and special occasions. 
            ''')

            # Product preferences and quality
            # st.subheader('Product preferences and quality:')
            # st.markdown('''
            # - Customers show a strong preference for specific product attributes such as "Double Stuffed Oreos," "Sweet Fillings," "Freshness and Taste," and "Good and Stuffed." This indicates the importance of understanding and catering to customer preferences.
            # - Topics related to packaging, such as "Shipping Damages," "Packaging and Quality," and "Crushed Cookies and Packaging," highlight the need for improving packaging to ensure products reach customers in excellent condition.
            # ''')

            # Customer engagement and feedback
            st.subheader('Customer engagement and feedback:')
            st.markdown('''
            - Topics like "Misleading Advertising" and "False Advertising" suggest that the business should ensure clear and accurate communication about product features to avoid negative customer experiences and feedback.
            - "Value for Money," "Great Value," and "Freshness and Price" indicate that customers are price-sensitive and appreciate products that offer good quality at reasonable prices.
            ''')

            # Product variety and innovation
            st.subheader('Customer preferences:')
            st.markdown('''
            - Topics like "Kids' Favorites," "Students and Taste", "Snacks ", and "Great for Lunch" show that different customer segments have specific preferences, and the business should consider tailoring its product offerings to cater to these diverse needs. 
            ''')

    # st.header("Sunburst Chart")
    # sunburst_chart = create_sunburst_chart()
    # st.plotly_chart(sunburst_chart)
    # st.header("Quadrant Tables")

    # data = {
    #    "Product": product_data,
    #    "Price": price_data,
    #    "Promotion": promotion_data,
    #    "Placement": placement_data
    # }

    # Pad the shorter columns with empty strings
    # max_length = max(len(product_data), len(price_data), len(promotion_data), len(placement_data))
    # for key, value in data.items():
    #    if len(value) < max_length:
    #        data[key] = value + [""] * (max_length - len(value))

    # single_table_df = pd.DataFrame(data)

    # st.write(single_table_df.style.set_table_attributes("style='display:inline'").set_caption("Single Table"))

with tab4:
    # st.header("4Ps Analysis")

    if product == 'B':
        st.subheader('Number of Topics per 4Ps for Product B')
        with open("product_B_piechart.html", "r") as f:
            html_content1 = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

        with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
            components.v1.html(html_content1, height=500, scrolling=False)

        st.subheader('Marketing Mix Tree for Product B')
        with open("new_B_tree.html", "r") as f:
            html_content = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

        with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
            components.v1.html(html_content, height=800, scrolling=True)

        st.subheader('4Ps Insights')
        pb1, pb2, pb3, pb4 = st.tabs(["Product", "Price", "Placement", "Promotion"])

        with pb1:
            # st.title("Product")
            st.write("""
            1. There is a wide variety of topics related to product attributes, such as taste, size, shape, quality, and flavors. This suggests that customers appreciate diversity in products, and the business should continue to innovate and introduce new flavors and combinations.
            2. The association with country of origin, culture and importance of quality is highlighted several times, emphasizing the value of maintaining and promoting the brand's origin and reputation. 
            3. Topics related to ingredients and allergens indicate the need for clear labeling and communication of product contents to cater to diverse customer needs and preferences.
            """)

        with pb2:
            st.title("Price")
            st.write("""
            1. "Price and Availability" and "Inflation and Economy" suggest that customers are price-sensitive and might be affected by economic factors. The business should consider offering a range of products at various price points to cater to different customer segments.
            2. "Great Purchase" implies that customers value deals and promotions, so the business should consider running special offers and discounts to attract and retain customers.
            """)

        with pb4:
            # st.title("Promotion")
            st.write("""
            1. A variety of promotional topics are mentioned, including events, celebrations, special occasions, and holidays. This suggests that the business should align its promotional activities with these events to create more targeted and effective marketing campaigns.
            2. Topics related to communication, engagement, and language highlight the importance of connecting with customers in a personalized and culturally relevant manner. This could involve using local languages in marketing materials and engaging with customers on social media platforms.
            3. The interest in branding and logo, Swiss identity, and brand history implies that the brand's story and heritage play a significant role in customer perception. The company should continue to emphasize its unique selling propositions and values in its promotional materials.
            """)

        with pb3:
            # st.title("Placement")
            st.write("""
            1. International topics, such as "European Market," "International Love," and "International Availability," indicate the potential for expanding the brand's global presence and catering to the needs of different markets.
            2. Topics like "Shipping and Delivery" and "Product Availability" underscore the importance of efficient distribution channels and making products easily accessible to customers.
            """)

    if product == 'A':
        st.subheader('Number of Topics per 4Ps for product A')
        with open("product_A_piechart.html", "r") as f:
            html_content1 = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

        with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
            components.v1.html(html_content1, height=500, scrolling=False)

        with open("new_A_tree.html", "r") as f:
            html_content = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)
        st.subheader('Marketing Mix Tree for Product A')
        with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
            components.v1.html(html_content, height=800, scrolling=True)

        st.subheader('4Ps Insights')
        pa1, pa2, pa3, pa4 = st.tabs(["Product", "Price", "Placement", "Promotion"])

        with pa1:
            # st.title("Product")
            st.write("""
            - Topics related to taste, such as "Snack quality," "Freshness and taste," and "Sweet fillings," emphasize the importance of delivering high-quality, flavourful products to satisfy customers.
            - The variety of products mentioned, such as "Long-lasting," "Double stuffing ," "Coffee flavor," and "Mint and new experiences," indicates a demand for diverse product offerings. The business should continue to innovate and introduce new flavors to keep customers engaged.
               """)

        with pa2:
            # st.title("Price")
            st.write("""
            - Topics like "Overpriced," "Value for money," "Great purchase," and "Great value" suggest that customers are price-sensitive and appreciate good deals on quality products. The business should consider offering competitive pricing and promotions to attract and retain customers.
            - The mention of "Double stuffing and price" and "Freshness and price" highlights specific products for which customers are particularly conscious of price. These products may require special attention to pricing strategies and value perception.
            """)

        with pa3:
            # st.title("Placement")
            st.write("""
            - "Shipping damages," "Freshness and damage," "Family and store," and "Fresh but broken" indicate that efficient and reliable distribution channels are crucial for an excellent customer experience. The business should optimize its shipping and distribution processes to ensure products are delivered promptly and in good condition.
            - The mention of "Irrelevant" placements suggests the need to assess and optimize product placement to ensure that products are available and easily accessible to customers in the right locations.
            """)

        with pa4:
            # st.title("Promotion")
            st.write("""
            - Topics related to promotional themes, such as "Kids' favorites," "Christmas treats," and "Birthday cakes and happiness," suggest that tailoring promotional activities to specific events, seasons, or customer segments can create more targeted and effective marketing campaigns.
            - Customer engagement topics like "Family time," "Love and snack variety," and "Humor and snack time" imply that connecting with customers on a personal level and creating a sense of community around the brand can positively impact customer loyalty and satisfaction.
            - Negative feedback on "False advertising" indicates the importance of accurate and transparent communication about product features in marketing materials to build trust and maintain customer satisfaction.
            """)

with tab5:
    select_box = st.multiselect('Choose Product', ['L', 'T', 'K'])

    # st.header('Competitive Analysis')

    # tab1, tab2, tab3 = st.tabs(['Sentiment Analysis','P Percentage', 'Subtopic Distribution'])

    # df = pd.read_csv(r'C:\Users\Arpan\Downloads\us_data_new.csv')
    st.info('This is an analysis of customer reviews across A and 3 of its competitors.')

    # image1 = Image.open('Oreo_logo_PNG1.png')
    # image2 = Image.open('images.png')
    # image3 = Image.open('images.jpg')
    # image4 = Image.open('images (1).png')
    # st.image([image1, image2, image3, image4], width=150)

    st.subheader('Total Data collected for different products')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("A", 19471)
    col2.metric("T", 2810)
    col3.metric("L", 6457)
    col4.metric("K", 3808)

    st.subheader('Mean Polarity across products')
    oreo = pd.read_csv('oreo_polarity.csv')
    TraderJoe = pd.read_csv('TraderJoe_quarterly_keywords_full.csv')
    Lotus = pd.read_csv('lotus_quarterly_keywords_full.csv')
    Keebler = pd.read_csv('keebler_quarterly_keywords_full.csv')

    scores = {'Product': ['A', 'T', 'L', 'K'],
              'Polarity Scores': [round(oreo['Oreo'].mean(), 2), round(TraderJoe['TraderJoe'].mean(), 2),
                                  round(Lotus['Lotus'].mean(), 2), round(Keebler['Keebler'].mean(), 2)]}
    scores_df = pd.DataFrame(scores)
    # st.write(scores_df)
    st.info(
        "Percentage of Positive, Neutral and Negative feedback across different products. We have selected Positive reviews with polarity score > 0.4, Neutral reviews within 0.2 to 0.4 & Negative reviews below 0.2 (Polarity scores are calculated using machine learning for sentiment analysis on text data. It ranges between -1 to 1). ")
    o_neu = round(len(oreo.loc[(oreo['Oreo'] >= 0.2) & (oreo['Oreo'] <= 0.4)]) / len(oreo) * 100, 1)
    o_neg = round(len(oreo.loc[(oreo['Oreo'] < 0.2)]) / len(oreo) * 100, 1)
    o_pos = round(len(oreo.loc[(oreo['Oreo'] > 0.4)]) / len(oreo) * 100, 1)

    t_neu = round(
        len(TraderJoe.loc[(TraderJoe['TraderJoe'] >= 0.2) & (TraderJoe['TraderJoe'] <= 0.4)]) / len(TraderJoe) * 100, 2)
    t_neg = round(len(TraderJoe.loc[(TraderJoe['TraderJoe'] < 0.2)]) / len(TraderJoe) * 100, 1)
    t_pos = round(len(TraderJoe.loc[(TraderJoe['TraderJoe'] > 0.4)]) / len(TraderJoe) * 100, 1)

    l_neu = round(len(Lotus.loc[(Lotus['Lotus'] >= 0.2) & (Lotus['Lotus'] <= 0.4)]) / len(Lotus) * 100, 1)
    l_neg = round(len(Lotus.loc[(Lotus['Lotus'] < 0.2)]) / len(Lotus) * 100, 1)
    l_pos = round(len(Lotus.loc[(Lotus['Lotus'] > 0.4)]) / len(Lotus) * 100, 1)

    k_neu = round(len(Keebler.loc[(Keebler['Keebler'] >= 0.2) & (Keebler['Keebler'] <= 0.4)]) / len(Keebler) * 100, 1)
    k_neg = round(len(Keebler.loc[(Keebler['Keebler'] < 0.2)]) / len(Keebler) * 100, 1)
    k_pos = round(len(Keebler.loc[(Keebler['Keebler'] > 0.4)]) / len(Keebler) * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    lst = ['Green', 'Orange', 'Red']
    with col1:
        st.subheader("A")
        emo_dict = {
            "Positive": str(o_pos) + "%",
            "Neutral": str(o_neu) + "%",
            "Negative": str(o_neg) + "%"
        }
        metrics = list(emo_dict.items())

        # Display each metric using st.metric
        for metric_key, metric_value in emo_dict.items():
            if metric_key == 'Negative':
                metric_value = f'<span style="color:red;">{metric_value}</span>'
            elif metric_key == 'Positive':
                metric_value = f'<span style="color:green;">{metric_value}</span>'
            elif metric_key == 'Neutral':
                metric_value = f'<span style="color:orange;">{metric_value}</span>'
            st.markdown(f'{metric_key}: {metric_value}', unsafe_allow_html=True)

    with col2:
        st.subheader("T")
        emo_dict = {
            "Positive": str(t_pos) + "%",
            "Neutral": str(t_neu) + "%",
            "Negative": str(t_neg) + "%"
        }
        metrics = list(emo_dict.items())

        # Display each metric using st.metric
        # color = "red" if "Metric 2" in metric_key else "green"  # Example: color code "Metric 2" key as red, others as green
        # st.markdown(f'<p style="color:{color};">{metric_key}</p>', unsafe_allow_html=True)
        # st.metric(metric_key, metric_value)

        for metric_key, metric_value in emo_dict.items():
            if metric_key == 'Negative':
                metric_value = f'<span style="color:red;">{metric_value}</span>'
            elif metric_key == 'Positive':
                metric_value = f'<span style="color:green;">{metric_value}</span>'
            elif metric_key == 'Neutral':
                metric_value = f'<span style="color:orange;">{metric_value}</span>'
            st.markdown(f'{metric_key}: {metric_value}', unsafe_allow_html=True)

    with col3:
        st.subheader("L")
        emo_dict = {
            "Positive": str(l_pos) + "%",
            "Neutral": str(l_neu) + "%",
            "Negative": str(l_neg) + "%"
        }
        metrics = list(emo_dict.items())

        # Display each metric using st.metric
        # for metric in metrics:
        #     st.metric(metric[0], metric[1])

        for metric_key, metric_value in emo_dict.items():
            if metric_key == 'Negative':
                metric_value = f'<span style="color:red;">{metric_value}</span>'
            elif metric_key == 'Positive':
                metric_value = f'<span style="color:green;">{metric_value}</span>'
            elif metric_key == 'Neutral':
                metric_value = f'<span style="color:orange;">{metric_value}</span>'
            st.markdown(f'{metric_key}: {metric_value}', unsafe_allow_html=True)
    with col4:
        st.subheader("K")
        emo_dict = {
            "Positive": str(k_pos) + "%",
            "Neutral": str(k_neu) + "%",
            "Negative": str(k_neg) + "%"
        }
        metrics = list(emo_dict.items())

        # Display each metric using st.metric
        # for metric in metrics:
        #     st.metric(metric[0], metric[1])

        for metric_key, metric_value in emo_dict.items():
            if metric_key == 'Negative':
                metric_value = f'<span style="color:red;">{metric_value}</span>'
            elif metric_key == 'Positive':
                metric_value = f'<span style="color:green;">{metric_value}</span>'
            elif metric_key == 'Neutral':
                metric_value = f'<span style="color:orange;">{metric_value}</span>'
            st.markdown(f'{metric_key}: {metric_value}', unsafe_allow_html=True)

    # image = Image.open('sunrise.jpg')
    #
    # st.image(image, caption='Sunrise by the mountains')
    st.info('The polarity ranges between -1 to 1 and is a measure of average sentiment where polarity<0 signifies negative \
        sentiment and positive polarity signifies a positive sentiment')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("A", scores['Polarity Scores'][0])
    col2.metric("T", scores['Polarity Scores'][1])
    col3.metric("L", scores['Polarity Scores'][2])
    col4.metric("K", scores['Polarity Scores'][3])

    st.subheader('Sentiment Across Quarters for Products Selected')

    cols = []
    for i in select_box:
        cols.append(i)
    # st.bar_chart(data=oreo,x='Quarter',y='polarity')

    # fig = px.bar(oreo, x='Quarter', y='polarity')
    # st.write(fig)

    # ax = oreo.plot(kind='bar', y='polarity', x='Quarter', figsize=(8, 6), rot=0)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig("mygraph.png")
    # plt.show()
    y_axis = ['Oreo']
    if len(cols) == 0:
        pass
    else:
        for c in cols:
            if c == 'L':
                oreo = oreo.merge(Lotus, on="Quarter")
                y_axis.append('Lotus')
            elif c == 'K':
                oreo = oreo.merge(Keebler, on="Quarter")
                y_axis.append('Keebler')
            else:
                oreo = oreo.merge(TraderJoe, on="Quarter")
                y_axis.append('TraderJoe')

    # st.write(oreo)
    fig = px.bar(oreo, x='Quarter', y=y_axis, barmode='group')
    fig.update_layout(
        autosize=False,
        width=1000,
        height=400,
        title="Polarity Distribution",
        xaxis_title="Quarter",
        yaxis_title="Polarity Scores",
        legend_title="Product"
    )
    newnames = {'Oreo': 'A', 'Keebler': 'K', 'TraderJoe': 'T', 'Lotus': 'L'}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                          )
                       )
    st.write(fig)
    # x=oreo.plot.bar()
    # st.write(x)

    # st.write(oreo)
    # st.bar_chart(data=oreo, x='Quarter', y=y_axis)
    # x=oreo.plot.bar()
    # st.pyplot(x)

    # # Example 3
    #
    st.info('This is an analysis of Mix Marketing model of A and 3 of its competitors. It involves proportion of reviews \
            related to the 4Ps namely: Product, Price, Promotion and Placement')

    st.subheader('Proportions across 4Ps')
    o = pd.read_excel('word_representation.xlsx', sheet_name=0)
    tj = pd.read_excel('compete.xlsx', sheet_name=3)
    lt = pd.read_excel('compete.xlsx', sheet_name=4)
    kb = pd.read_excel('compete.xlsx', sheet_name=5)

    dict_oreo = {'Item': 'A',
                 'Product': float(o["4P's"].value_counts()['Product'] / o["4P's"].value_counts().sum()) * 100,
                 'Price': float(o["4P's"].value_counts()['Price'] / o["4P's"].value_counts().sum()) * 100,
                 'Promotion': float(o["4P's"].value_counts()['Promotion'] / o["4P's"].value_counts().sum()) * 100,
                 'Placement': float(o["4P's"].value_counts()['Placement'] / o["4P's"].value_counts().sum()) * 100}
    df = pd.DataFrame(dict_oreo, index=[0])
    x = ['Product', 'Price', 'Promotion', 'Placement']
    data = []
    data.append(go.Bar(name='A', x=x, y=[df.loc[len(df.index) - 1]['Product'], df.loc[len(df.index) - 1]['Price'],
                                         df.loc[len(df.index) - 1]['Promotion'],
                                         df.loc[len(df.index) - 1]['Placement']]))
    if len(cols) == 0:
        pass
    else:
        for c in cols:
            if c == 'L':
                df.loc[len(df.index)] = ['L', float(lt.nunique()['Product'] / lt.nunique().sum()) * 100,
                                         float(lt.nunique()['Price'] / lt.nunique().sum()) * 100,
                                         float(lt.nunique()['Promotion'] / lt.nunique().sum()) * 100,
                                         float(lt.nunique()['Placement'] / lt.nunique().sum()) * 100]
                a = go.Bar(name='L', x=x,
                           y=[df.loc[len(df.index) - 1]['Product'], df.loc[len(df.index) - 1]['Price'],
                              df.loc[len(df.index) - 1]['Promotion'], df.loc[len(df.index) - 1]['Placement']])
                data.append(a)
            elif c == 'K':
                df.loc[len(df.index)] = ['K', float(kb.nunique()['Product'] / kb.nunique().sum()) * 100,
                                         float(kb.nunique()['Price'] / kb.nunique().sum()) * 100,
                                         float(kb.nunique()['Promotion'] / kb.nunique().sum()) * 100,
                                         float(kb.nunique()['Placement'] / kb.nunique().sum()) * 100]
                b = go.Bar(name='K', x=x,
                           y=[df.loc[len(df.index) - 1]['Product'], df.loc[len(df.index) - 1]['Price'],
                              df.loc[len(df.index) - 1]['Promotion'], df.loc[len(df.index) - 1]['Placement']])
                data.append(b)
            else:
                df.loc[len(df.index)] = ['T', float(tj.nunique()['Product'] / tj.nunique().sum()) * 100,
                                         float(tj.nunique()['Price'] / tj.nunique().sum()) * 100,
                                         float(tj.nunique()['Promotion'] / tj.nunique().sum()) * 100,
                                         float(tj.nunique()['Placement'] / tj.nunique().sum()) * 100]
                c = go.Bar(name='T', x=x,
                           y=[df.loc[len(df.index) - 1]['Product'], df.loc[len(df.index) - 1]['Price'],
                              df.loc[len(df.index) - 1]['Promotion'],
                              df.loc[len(df.index) - 1]['Placement']])
                data.append(c)
    # st.write(df)
    # st.write(data)
    fig = go.Figure(data)
    fig.update_layout(barmode='group', yaxis_range=[0, 100])
    fig.update_layout(
        autosize=False,
        width=1000,
        height=400,
        title="Proportion of P's",
        yaxis_title="Percentage Share",
        legend_title="Product"
    )
    st.plotly_chart(fig)

    st.info('This analyses different subtopics present across the 4Ps among A and its competitors.')

    # a1 = pd.read_csv('all_oreo_data.csv')

    # col1, col2, col3, col4 = st.columns(4)
    # col1.metric("Oreo", st.image(image1, width=10))
    # col2.metric("TraderJoe", st.image(image2, width=10))
    # col3.metric("Lotus", st.image(image3, width=10))
    # col4.metric("Keebler", st.image(image4, width=10))

    st.subheader('Subtopics across certain P')

    ore = pd.read_excel('compete.xlsx', sheet_name=6)
    select_p = st.selectbox('Choose P', ['Product', 'Price', 'Promotion', 'Placement'])
    val = []
    p_dict = {}
    for i in select_p:
        val.append(i)
    print(val)
    if len(val) == 0:
        pass
    else:
        temp = ''.join(str(i) for i in val)
        print(temp)
        if temp == 'Product':
            p_dict = {'A': ore['Product'],
                      'L': lt['Product'],
                      'K': kb['Product'],
                      'T': tj['Product']}

            val_df = pd.DataFrame(p_dict)
            val_df.dropna(axis=0, how='all', inplace=True)
            st.write(val_df)
        elif temp == 'Price':
            p_dict = {'A': ore['Price'],
                      'L': lt['Price'],
                      'K': kb['Price'],
                      'T': tj['Price']}

            val_df = pd.DataFrame(p_dict)
            val_df.dropna(axis=0, how='all', inplace=True)
            st.write(val_df)
        elif temp == 'Promotion':
            p_dict = {'A': ore['Promotion'],
                      'L': lt['Promotion'],
                      'K': kb['Promotion'],
                      'T': tj['Promotion']}

            val_df = pd.DataFrame(p_dict)
            val_df.dropna(axis=0, how='all', inplace=True)
            st.write(val_df)
        else:
            p_dict = {'A': ore['Placement'],
                      'L': lt['Placement'],
                      'K': kb['Placement'],
                      'T': tj['Placement']}

            val_df = pd.DataFrame(p_dict)
            val_df.dropna(axis=0, how='all', inplace=True)
            st.write(val_df)

# with tab6:
#     # select_box = st.multiselect('Choose Product you want to query about', ['A','B' 'L', 'T', 'K'])

#     st.info("Welcome to our chatbot. Happy texting! 🤗")
#     # Build prompt
#     os.environ['OPENAI_API_KEY'] = "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"

#     def conversational_chat(query):
#         question = query
#         result = chain({"query": question,
#                         "chat_history": st.session_state['history']})
#         st.session_state['history'].append((query, result["result"]))
#         return result["result"]

#     loaders = [CSVLoader(file_path='mergedata-AB.csv', encoding="utf-8", csv_args={'delimiter': ','}),
#         CSVLoader(file_path='marketing.csv', encoding="utf-8", csv_args={'delimiter': ','})]


#     data = []
#     for loader in loaders:
#         data.extend(loader.load())

#     # st.write("data read")
#     from langchain.text_splitter import RecursiveCharacterTextSplitter

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )

#     splits = text_splitter.split_documents(data)
#     embeddings = OpenAIEmbeddings()
#     # openai_api_key = "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"
#     persist_directory = 'docs/chroma/'
#     vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,persist_directory=persist_directory)
#     # vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#     llm = ChatOpenAI(temperature=0.0, model_name='gpt-4')


#     # chain = RetrievalQA.from_chain_type(
#     #     llm,
#     #     retriever=vectorstore.as_retriever(),
#     #     return_source_documents=True,
#     #     chain_type="refine"
#     # )


#     template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
#             {context}
#             Question: {question}
#             Helpful Answer:"""
#     QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#     # chain = ConversationalRetrievalChain.from_llm(
#     #     llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', memory=memory),
#     #     retriever=vectorstore.as_retriever())

#     chain = RetrievalQA.from_chain_type(
#         llm,
#         retriever=vectorstore.as_retriever(),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )




#     # st.write("2nd step")

#     if 'history' not in st.session_state:
#         st.session_state['history'] = []

#     if 'generated' not in st.session_state:
#         st.session_state['generated'] = ["Hello ! Ask me anything about " + " 🤗"]

#     if 'past' not in st.session_state:
#         st.session_state['past'] = ["Hey ! 👋"]

#         # container for the chat history
#     response_container = st.container()
#     # container for the user's text input
#     container = st.container()

#     with container:
#         with st.form(key='my_form', clear_on_submit=True):
#             user_text = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
#             submit_button = st.form_submit_button(label='Send')
#         if submit_button and user_text:
#             output = conversational_chat(user_text)
#             st.session_state['past'].append(user_text)
#             st.session_state['generated'].append(output)

#     if st.session_state['generated']:
#         with response_container:
#             for i in range(len(st.session_state['generated'])):
#                 message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
#                 message(st.session_state["generated"][i], key=str(i))
# # , avatar_style="big-smile"
# # , avatar_style="thumbs"



