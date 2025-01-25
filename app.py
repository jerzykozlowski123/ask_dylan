import streamlit as st
from dotenv import dotenv_values, load_dotenv
from qdrant_client import QdrantClient
import streamlit.components.v1 as components
from langfuse.decorators import observe
from langfuse.openai import OpenAI
import os

st.set_page_config(page_title="Zapytaj Boba Dylana", layout="centered", menu_items={'About': 'Zaytaj Boba Dylana by JK'})

model_pricings = {
    "gpt-4o": {
        "input_tokens": 2.50 / 1_000_000,  # per token
        "output_tokens": 10.00 / 1_000_000,  # per token
    },
    "gpt-4o-mini": {
        "input_tokens": 0.15 / 1_000_000,  # per token
        "output_tokens": 0.60 / 1_000_000,  # per token
    },
    'text-embedding-ada-002': {
        "total_tokens": 0.40 / 1_000_000,  # per token
    }
}
MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = 'text-embedding-ada-002'
USD_TO_PLN = 4.02
PRICING = model_pricings[MODEL]
EMBEDDING_DIM = 1536
QDRANT_COLLECTION_NAME = 'dylans_songs'
session_token_limit = 10_000

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Spytaj Boba Dylana")

def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
    )

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        st.write('Błąd połączenia z kolekcją wektorów.')

@observe(name="response") 
def chatbot_reply(user_prompt, memory):

    if st.session_state['total_tokens_used'] >= session_token_limit:
        return {
            "role": "assistant",
            "content": "Przekroczono limit tokenów na sesję.",
            "usage": {}
        }
    # dodaj system message
    messages = [
        {
            'role': 'system',
            'content': st.session_state['chatbot_personality'],
        },
    ]
    # dodaj wszystkie wiadomości z pamięci
    for message in memory:
        messages.append({'role': message['role'], 'content': message['content']})
    
    # dodaj wiadomość użytkownika
    messages.append({'role': 'user', 'content': user_prompt})

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    usage = {}
    if response.usage:
        usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        st.session_state['total_tokens_used'] += response.usage.total_tokens
    return {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "usage": usage,
    }

@observe(name="create_embedding") 
def get_embedding(text):
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
    )
    embedding = result.data[0].embedding
    tokens_used = result.usage.total_tokens # Pobranie liczby tokenów
    st.session_state['total_tokens_used'] += tokens_used
    return embedding, tokens_used

@observe(name="find_embedding") 
def search_similar_song(embedding):
    qdrant_client = get_qdrant_client()
    response = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=embedding,
        limit=1  # Zwrócenie tylko najbardziej dopasowanej piosenki
    )
    if response:
        lyrics = response[0].payload['lyrics']
        title = response[0].payload['title']
        similarity_score = response[0].score 
        return lyrics, title, similarity_score
    return None, None

tokens_used = 0
if 'total_tokens_used' not in st.session_state:
    st.session_state['total_tokens_used'] = 0
if "messages" not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('O co chcesz zapytać?', max_chars=250)
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state["messages"].append({"role": "user", "content": prompt})

    embedding, tokens_used = get_embedding(prompt)
    lyrics, title, similarity_score = search_similar_song(embedding)

    if lyrics and title:
        additional_prompt = f"Bob Dylan, tytuł: {title}, tekst piosenki: {lyrics}"
    else:
        additional_prompt = ""
    with st.chat_message("assistant"):
        response = chatbot_reply(user_prompt=prompt + " " + additional_prompt, memory=st.session_state["messages"][-3:])
        # similarity_message = f"\n\n**Similarity score**: {similarity_score:.4f}"
        st.markdown(response["content"]) # + similarity_message

    st.session_state["messages"].append({"role": "assistant", "content": response["content"], "usage": response["usage"]})

with st.sidebar:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.image("./pictures/Bob_Dylan_logo_l.png")
    st.title("Zapytaj Boba Dylana")
    # st.write("Aktualne modele AI:")
    # st.write(f"{MODEL}, {EMBEDDING_MODEL}")

    total_cost = 0
    for message in st.session_state["messages"]:
        if "usage" in message:
            total_cost += message["usage"]["prompt_tokens"] * PRICING["input_tokens"]
            total_cost += message["usage"]["completion_tokens"] * PRICING["output_tokens"]
    
    embedding_cost = tokens_used * model_pricings['text-embedding-ada-002']["total_tokens"]
    total_cost += embedding_cost

    # c0, c1 = st.columns(2)
    # with c0:
    #     st.metric("Koszt rozmowy (USD)", f"${total_cost:.4f}")

    # with c1:
    #     st.metric("Koszt rozmowy (PLN)", f"{total_cost * USD_TO_PLN:.4f}")

    buy_me_a_coffee_button = """ <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="jerzykozlowski" data-color="#FFDD00" data-emoji="☕" data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff"></script> """
    components.html(buy_me_a_coffee_button, height=70)

    st.session_state['chatbot_personality'] = """
Jesteś znawcą twórczości Boba Dylana, który odpowiada na wszystkie pytania użytkownika,  
wplatając w odpowiedź odpowiedni do kontekstu cytat z dołączonej piosenki, podanym tytułem i autorem. 
Odpowiadaj na pytania w sposób zwięzły i zrozumiały.
        """
    st.write("Tłumaczenia pochodzą z https://www.groove.pl/")

    # Footer
    footer_style = """
        <style>
            .footer {
                bottom: 0;
                left: 0;
                right: 0;
                background-color: transparent;
                text-align: center;
                padding: 10px;
                font-size: 14px;
                border-top: 1px solid #e7e7e7;
                color: inherit;
            }
            body {
                padding-bottom: 50px;
            }
        </style>
    """

    footer_html = """
    <div class="footer">
        <p>Contact: Jerzy Kozlowski | <a href="mailto:jerzykozlowski@mailmix.pl">jerzykozlowski@mailmix.pl</a></p>
    </div>
    """

    st.markdown(footer_style, unsafe_allow_html=True)
    st.markdown(footer_html, unsafe_allow_html=True)