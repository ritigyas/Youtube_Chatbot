# YouTube Transcript Chatbot

## Overview

This project is a premium Streamlit web application that allows users to chat with any English YouTube video using Retrieval-Augmented Generation (RAG).

The application automatically extracts the transcript of a YouTube video, converts the content into embeddings, stores them in a FAISS vector database, and enables users to ask natural language questions about the video's content.

### Features

* Modern Streamlit UI with dark theme and glassmorphism design
* Automatic YouTube transcript extraction
* Intelligent text chunking using LangChain
* Semantic search using FAISS vector database
* Gemini Embeddings (`gemini-embedding-001`) for vector generation
* Gemini 2.5 Flash for conversational question answering
* Secure API key management using environment variables
* Real-time chat interface
* Fast and scalable retrieval pipeline

---

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Google Gemini API
* YouTube Transcript API
* Python Dotenv

---

## Project Architecture

YouTube Video URL

↓

Transcript Extraction

↓

Text Chunking

↓

Gemini Embeddings

↓

FAISS Vector Store

↓

Retriever

↓

Gemini 2.5 Flash

↓

Answer Generation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/youtubeChatbot.git
cd youtubeChatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

---

## Running the Application

```bash
streamlit run app.py
```

After launching:

1. Open the Streamlit application in your browser.
2. Paste a YouTube video URL.
3. Click **Ingest Video**.
4. Wait for indexing to complete.
5. Start asking questions about the video.

---

## Demo Video

Upload a screen recording of your application and replace the link below:

**Demo Video:**

https://drive.google.com/file/d/1-qzu9MQp5snwwtCJLvcruB8MldkR9Pwg/view?usp=sharing



## Screenshots

### Home Page

Add screenshot here:
<img width="1917" height="971" alt="image" src="https://github.com/user-attachments/assets/c0a34c4b-b5d1-4115-8218-90ad07f98cd7" />


### Chat Interface

<img width="1917" height="972" alt="image" src="https://github.com/user-attachments/assets/7bb71317-61bb-4024-8248-4afe64d93588" />


## Folder Structure

```text
youtubeChatbot/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
│
├── screenshots/
│   ├── home.png
│   └── chat.png
│
└── .streamlit/
    └── config.toml
```

---

## Future Improvements

* Multi-language transcript support
* Chat history persistence
* PDF export of conversations
* Support for playlists
* Hybrid retrieval techniques
* Source citation highlighting

---

## Contributing

Contributions are welcome.

Feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.
