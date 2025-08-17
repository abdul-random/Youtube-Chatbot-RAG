import os
import re
import pathlib
import streamlit as st

from youtube_transcript_api import (
    YouTubeTranscriptApi, TranscriptsDisabled,
    NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript
)

from langchain.schema import Document
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import messages_from_dict

openai_api_key = os.getenv("OPENAI_API_KEY")

# ---------- Utils ----------
def to_lc_format(messages):
    role_map = {"system": "system", "user": "human", "assistant": "ai"}
    return [
        {"type": role_map[m["role"]], "data": {"content": m["content"], "additional_kwargs": {}}}
        for m in messages
    ]

def extract_video_id(youtube_url: str) -> str:
    m = re.search(r"(?:v=|be/|embed/)([A-Za-z0-9_-]{11})", youtube_url)
    if not m:
        # fallback to original pattern
        m = re.search(r"watch\?v=([^#&]+)", youtube_url)
    if not m:
        raise ValueError("Could not find a valid YouTube video id in the URL.")
    vid = m.group(1)
    vid = vid.split("&")[0]
    if not vid:
        raise ValueError("Could not extract a valid video_id from the URL.")
    return vid

def combine_text(docs):
    return " ".join([doc.page_content for doc in docs])

def format_chat_history(messages, max_chars=4000):
    """Return last messages (user/assistant) as a single string, trimmed to max_chars."""
    chunks = []
    for m in messages:
        if m["role"] == "system":
            continue
        prefix = "User:" if m["role"] == "user" else "Assistant:"
        chunks.append(f"{prefix} {m['content']}")
    text = "\n".join(chunks[-20:])  # keep last 20 exchanges
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text.strip()

def thumbnail_url(video_id: str) -> str:
    # hqdefault is usually fine; fallback options: maxresdefault, mqdefault
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

def faiss_path_for(video_id: str) -> str:
    base = pathlib.Path("faiss_indexes")
    base.mkdir(parents=True, exist_ok=True)
    return str(base / video_id)

# ---------- Models ----------
llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.2, api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

# ---------- UI ----------
st.sidebar.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
    }
    .sidebar-logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80px;  /* adjust logo size */
        margin-bottom: 8px;
    }
    </style>
    <img class="sidebar-logo" src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png" alt="YouTube Bot Logo">
    <div class="sidebar-title">YouTube Chatbot</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    h1.shiny-title {
        font-size: 36px !important;   /* force smaller size */
        font-weight: 700;
        text-align: center;
        background: linear-gradient(270deg, #ff6ec7, #845ec2, #ff9671, #f9f871);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientFlow 8s ease infinite;
        margin: 0.25rem 0 1rem;
    }
    </style>
    <h1 class="shiny-title">✨ Ask me anything about the video ✨</h1>
    """,
    unsafe_allow_html=True
)

# Session state bootstrapping
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful YouTube bot."}]

if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "video_link": "",
        "selected_language": "",
        "raw_transcript": "",
        "translated_transcript": "",
        "status_message": "",
        "video_id": "",
    }

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------- Sidebar ----------
st.sidebar.header("Insert YouTube Video Link")
text_input = st.sidebar.text_input("Video Link:", value=st.session_state.inputs.get("video_link", ""))
dropdown_choice = st.sidebar.selectbox("Choose a Language:", ["English", "Hindi"], index=0)
submitted = st.sidebar.button("Submit")

# Preview container (thumbnail/video)
preview_box = st.sidebar.container()

# ---------- Submit Handler ----------
if submitted:
    # Reset chat history on submit
    st.session_state.messages = [{"role": "system", "content": "You are a helpful YouTube bot."}]
    # Reset inputs & vector
    st.session_state.inputs.update({
        "video_link": text_input,
        "selected_language": dropdown_choice,
        "raw_transcript": "",
        "translated_transcript": "",
        "status_message": "",
        "video_id": "",
    })
    st.session_state.vector_store = None

    if not text_input.strip():
        st.session_state.inputs["status_message"] = "⚠️ Please enter a valid YouTube video link."
    else:
        try:
            video_id = extract_video_id(text_input)
            st.session_state.inputs["video_id"] = video_id

            # Show video if possible; otherwise thumbnail
            try:
                preview_box.video(f"https://www.youtube.com/watch?v={video_id}")
            except Exception:
                preview_box.image(thumbnail_url(video_id), caption="YouTube Thumbnail")

            # Loading indicator while vectors are created
            with st.spinner("Fetching transcript and building vectors…"):
                # Fetch transcript
                yt_ins = YouTubeTranscriptApi()
                transcript_list = yt_ins.list(video_id = video_id)

                # Language preference handling
                lang_map = {"English": "en", "Hindi": "hi"}
                lang_code = lang_map.get(dropdown_choice, "en")

                transcript = transcript_list.find_transcript([lang_code])
                txt_snippets = transcript.fetch()

                combined_text = combined_text = " ".join([s.text for s in txt_snippets.snippets])
                st.session_state.inputs["raw_transcript"] = combined_text

                # Translate if not English
                if lang_code != "en":
                    translation_prompt = (
                        "Translate the following text into English without omitting any part.\n\n"
                        f"{combined_text}"
                    )
                    translated = llm.invoke(translation_prompt).content
                else:
                    translated = combined_text

                st.session_state.inputs["translated_transcript"] = translated

                # Build or reuse FAISS index (persisted locally per video)
                index_dir = faiss_path_for(video_id)
                if pathlib.Path(index_dir).exists():
                    # Reuse on-disk index for the same video_id
                    st.session_state.vector_store = FAISS.load_local(
                        index_dir, embeddings, allow_dangerous_deserialization=True
                    )
                else:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = splitter.create_documents([translated])
                    vs = FAISS.from_documents(docs, embeddings)
                    vs.save_local(index_dir)
                    st.session_state.vector_store = vs

                st.session_state.inputs["status_message"] = f"✅ Ready! Transcript processed ({dropdown_choice})."

        except ValueError as e:
            st.session_state.inputs["status_message"] = f"⚠️ {e}"
        except TranscriptsDisabled:
            st.session_state.inputs["status_message"] = "❌ Captions are disabled for this video."
        except NoTranscriptFound:
            st.session_state.inputs["status_message"] = "❌ No transcript found for the requested language."
        except VideoUnavailable:
            st.session_state.inputs["status_message"] = "❌ The video is unavailable (removed, private, or region-restricted)."
        except CouldNotRetrieveTranscript:
            st.session_state.inputs["status_message"] = "⚠️ Could not retrieve transcript due to a YouTube issue."
        except Exception as e:
            st.session_state.inputs["status_message"] = f"⚠️ Unexpected error: {e}"

# Status message (persistent)
if st.session_state.inputs["status_message"]:
    st.sidebar.info(st.session_state.inputs["status_message"])

# If we have a video id and no video widget rendered yet (e.g., on rerun), show a thumbnail
if st.session_state.inputs["video_id"] and not submitted:
    try:
        preview_box.video(f"https://www.youtube.com/watch?v={st.session_state.inputs['video_id']}")
    except Exception:
        preview_box.image(thumbnail_url(st.session_state.inputs["video_id"]), caption="YouTube Thumbnail")

# ---------- Chat Area ----------
# Freeze chat until vectors exist
chat_disabled = st.session_state.vector_store is None
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_msg = st.chat_input(
    "Ask about the video…",
    disabled=chat_disabled
)

# ---------- Chat Handler ----------
if user_msg and not chat_disabled:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.chat_message("user").markdown(user_msg)

    # Retriever
    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "lambda_mult": 0.9}
    )
    parser = StrOutputParser()

    # Parallel: get query passthrough and retrieve context
    retriever_chain = RunnableParallel(
        {
            "query": RunnablePassthrough(),
            "context": retriever | RunnableLambda(combine_text)
        }
    )

    # Prompt that prefers context first, then chat history, else "I don't know."
    qa_template = PromptTemplate(
        template=("You are a helpful assistant.\n\n"
        "Context (from the video transcript):\n{context}\n\n"
        "Chat history (latest first):\n{chat_history}\n\n"
        "User query:\n{query}\n\n"
        "Instructions:\n"
        "1) First, answer using ONLY the provided Context.\n"
        "2) If the answer is not fully available in Context, use Chat history to fill gaps.\n"
        "3) If neither Context nor Chat history is sufficient:\n"
        "   - If the query has some relevance, provide a thoughtful answer using your own knowledge, "
        "but stay strictly factual and do not hallucinate.\n"
        "   - If the query is completely unrelated, reply exactly: \"I don't know.\"\n\n"
        "Be concise and directly address the query."),
        input_variables=["context", "chat_history", "query"]
        )

    # Compose chain: add chat history before LLM
    def add_chat_history(inputs):
        return {
            "context": inputs["context"],
            "query": inputs["query"],
            "chat_history": format_chat_history(st.session_state.messages)
        }

    chain = retriever_chain | RunnableLambda(add_chat_history) | qa_template | llm | parser

    # Invoke chain with user query
    answer = chain.invoke(user_msg)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------- Debug (optional) ----------
# st.write("STATE INPUTS:", st.session_state.inputs)
# st.write("Has Vector Store:", st.session_state.vector_store is not None)
