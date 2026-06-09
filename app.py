import streamlit as st
import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Page configurations
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Inject Custom CSS for Rich Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* App background & fonts */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0F0F14 0%, #15151E 100%);
    }
    
    /* Sleek scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0F0F14;
    }
    ::-webkit-scrollbar-thumb {
        background: #FF003C;
        border-radius: 4px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E1E24 !important;
        border-right: 1px solid rgba(255, 0, 60, 0.2);
    }
    
    /* Glassmorphic header cards */
    .header-card {
        background: rgba(30, 30, 36, 0.7);
        border: 1px solid rgba(255, 0, 60, 0.3);
        box-shadow: 0 8px 32px 0 rgba(255, 0, 60, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        text-align: center;
    }
    
    .header-title {
        background: linear-gradient(90deg, #FFFFFF 0%, #FF003C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        color: #B0B0C0;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Features cards */
    .feature-card {
        background: rgba(30, 30, 36, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 16px;
        margin-top: 16px;
        transition: transform 0.2s ease, border 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 0, 60, 0.4);
    }
    
    /* Custom buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #E50914 0%, #FF003C 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 0, 60, 0.3) !important;
        width: 100% !important;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(255, 0, 60, 0.5) !important;
    }
    
    /* Ingest status box */
    .status-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
        margin-bottom: 16px;
    }
    
    /* Custom system alerts */
    .custom-alert {
        background-color: rgba(255, 0, 60, 0.1);
        border-left: 4px solid #FF003C;
        color: #FFF;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to extract YouTube Video ID
def get_video_id(url):
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

# Helper function to fetch transcript text
def fetch_transcript_text(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        # Gracefully handle both dict and object structures
        text = " ".join(
            getattr(snippet, 'text', snippet.get('text') if isinstance(snippet, dict) else "")
            for snippet in fetched
        )
        return text, None
    except TranscriptsDisabled:
        return None, "Subtitles/transcripts are disabled for this video."
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"

# Helper function to build Vector Store
@st.cache_resource(show_spinner=False)
def build_vector_store(transcript_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Helper function to generate response
def generate_response(vector_store, question, chat_history):
    # Retrieve relevant documents
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format chat history
    history_str = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
        
    system_prompt = f"""You are an intelligent AI assistant trained to answer questions about a YouTube video based on its transcript.
    
Use the following context from the video transcript to answer the user's question. If you do not know the answer or if the context does not contain the answer, say clearly that you cannot find the answer in the video transcript. Be helpful, concise, and accurate.

Context:
{context}

Conversation History:
{history_str}
User's Question: {question}

Answer:"""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response = llm.invoke(system_prompt)
    return response.content

# Initialize Session State Variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None

# Header Banner
st.markdown("""
<div class="header-card">
    <h1 class="header-title">YouTube Transcript Chatbot</h1>
    <p class="header-subtitle">Analyze transcripts, extract insights, and chat with any YouTube video in real-time</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("<h2 style='color: #FF003C; font-weight: 700; margin-bottom: 20px;'>Control Panel</h2>", unsafe_allow_html=True)
    
    # API Key Handling
    api_key_status = "Loaded from .env" if os.getenv("GEMINI_API_KEY") else "Not configured"
    st.info(f"🔑 Gemini API Key: **{api_key_status}**")
    
    # Video Input Section
    video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", help="Enter any English YouTube video link")
    
    ingest_button = st.button("🚀 Process & Ingest Video")
    
    if ingest_button:
        if not video_url:
            st.error("Please enter a valid YouTube video URL.")
        elif not os.getenv("GEMINI_API_KEY"):
            st.error("API Key not found. Please verify your .env file contains GEMINI_API_KEY.")
        else:
            video_id = get_video_id(video_url)
            if not video_id:
                st.error("Could not parse YouTube Video ID. Check the URL structure.")
            else:
                with st.spinner("Fetching transcript and building index..."):
                    transcript_text, err = fetch_transcript_text(video_id)
                    if err:
                        st.error(err)
                    else:
                        vector_store = build_vector_store(transcript_text)
                        
                        # Reset history on new ingestion
                        st.session_state.vector_store = vector_store
                        st.session_state.current_video_id = video_id
                        st.session_state.transcript = transcript_text
                        st.session_state.chat_history = []
                        st.success("Video successfully processed and indexed!")
                        
    # Video Details Panel
    if st.session_state.current_video_id:
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #FF003C; font-weight: 600;'>Active Video</h4>", unsafe_allow_html=True)
        
        # Display Video Thumbnail
        st.image(f"https://img.youtube.com/vi/{st.session_state.current_video_id}/0.jpg", use_column_width=True)
        st.write(f"🔗 Video ID: `{st.session_state.current_video_id}`")
        
        # Expandable Transcript Preview
        with st.expander("📝 View Transcript Preview"):
            st.write(st.session_state.transcript[:1000] + "...")

# Main Workspace
if st.session_state.vector_store is None:
    # Onboarding view
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <h3 style="color: #FF003C; font-weight: 600;">⚡ Instant Transcript Retrieval</h3>
            <p style="color: #B0B0C0; font-size: 0.95rem;">Fetches transcripts automatically via YouTube API without manual copy-pasting.</p>
        </div>
        <div class="feature-card">
            <h3 style="color: #FF003C; font-weight: 600;">🧠 Semantic Chunk Indexing</h3>
            <p style="color: #B0B0C0; font-size: 0.95rem;">Segments transcript text and embeds them into a local vector store using gemini-embedding-001.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <h3 style="color: #FF003C; font-weight: 600;">🔍 RAG-Powered Conversational AI</h3>
            <p style="color: #B0B0C0; font-size: 0.95rem;">Answers your queries directly referencing text passages for high-fidelity responses.</p>
        </div>
        <div class="feature-card">
            <h3 style="color: #FF003C; font-weight: 600;">💬 Multi-Turn Dialogues</h3>
            <p style="color: #B0B0C0; font-size: 0.95rem;">Remembers the conversation context to provide smooth follow-up answers.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='text-align: center; margin-top: 40px; color: #808090;'>👈 Input a YouTube URL in the sidebar to start chatting!</div>", unsafe_allow_html=True)

else:
    # Chat interface
    st.markdown(f"<h3 style='color: #FFF; font-weight: 600; margin-bottom: 20px;'>Chatting with Video `{st.session_state.current_video_id}`</h3>", unsafe_allow_html=True)
    
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    # Input box
    if user_question := st.chat_input("Ask a question about the video..."):
        # Add to history and display
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing transcript..."):
                response_text = generate_response(
                    st.session_state.vector_store,
                    user_question,
                    st.session_state.chat_history[:-1]
                )
                st.write(response_text)
                
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
