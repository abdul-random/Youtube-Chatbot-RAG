# YouTube Chatbot 🎥🤖

A Streamlit-based chatbot that allows you to **ask questions about any YouTube video** using its transcript. It applies a **Retrieval-Augmented Generation (RAG)** pipeline with OpenAI models and FAISS for semantic search.

---

## 🚀 Features

* Extracts transcripts from YouTube videos in **English** or **Hindi**.
* Optionally translates non-English transcripts into English.
* Stores and retrieves transcript data using **FAISS vector search**.
* Provides **context-aware answers** by combining:

  1. Transcript context
  2. Chat history
  3. Model knowledge (if needed)
* Interactive **Streamlit chat UI** with video preview.

---

## 📦 Installation

1. Clone the repository:

```bash
 git clone https://github.com/yourusername/youtube-chatbot.git
 cd youtube-chatbot
```

2. Create a virtual environment and install dependencies:

```bash
 python -m venv venv
 source venv/bin/activate   # On Windows: venv\Scripts\activate
 pip install -r requirements.txt
```

3. Add your OpenAI API key:

```bash
 export OPENAI_API_KEY="your_api_key_here"   # Linux/Mac
 setx OPENAI_API_KEY "your_api_key_here"     # Windows (Powershell)
```

4. Run the app:

```bash
 streamlit run app.py
```

---

## 🧩 RAG Process Flow

```
User Query
   │
   ▼
FAISS Retriever ──► Context
        │              │
        └── Chat History
               │
               ▼
        Prompt Template
               │
               ▼
               LLM
               │
               ▼
             Answer
```

**Steps:**

1. User enters a query.
2. FAISS retriever fetches the most relevant transcript chunks.
3. Chat history is formatted and appended.
4. Both context and history are passed to the prompt template.
5. The LLM generates a concise, factual answer.

**Troubleshooting (Mermaid not rendering):**

* **GitHub.com:** Works in the normal README view (not in **Raw** view). Refresh the page if it stalls.
* **VS Code:** Install the **"Markdown Preview Mermaid Support"** extension (or view on GitHub).
* **GitHub Enterprise / other platforms:** Mermaid may be disabled by admins or unsupported. In that case, keep the text fallback or add a static image of the diagram.
* **PyPI / Streamlit / other renderers:** Most do not render Mermaid. Use the text fallback or include a PNG/SVG instead.

---

## 🛠️ Tech Stack

* [Streamlit](https://streamlit.io/) – Web app framework
* [LangChain](https://www.langchain.com/) – Orchestration
* [OpenAI](https://platform.openai.com/) – LLM & embeddings
* [FAISS](https://github.com/facebookresearch/faiss) – Vector store
* [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/) – Transcript extraction

---

## 📂 Project Structure

```
youtube-chatbot/
│── app.py                # Main Streamlit app
│── requirements.txt      # Python dependencies
│── faiss_indexes/        # Saved FAISS vector indexes (per video)
│── README.md             # Documentation (this file)
```

---

## ⚠️ Limitations

* Some videos may have **captions disabled** → no transcript available.
* Translation quality depends on LLM.
* Requires OpenAI API key (paid usage may apply).

---

## 📌 Future Improvements

* Add support for **more languages**.
* Enable **summarization mode**.
* Store **conversation memory** per video.

---

## 👩‍💻 Author

Built with ❤️ using Streamlit + LangChain + OpenAI.

Abdul Khalee Shaik
