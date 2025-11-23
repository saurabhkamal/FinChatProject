# app.py  — FinChat Pro with:
# - PDF RAG via Pinecone
# - Financial website crawling (Macrotrends, StockAnalysis, CompaniesMarketCap)
# - Chroma vector store for CSV + crawled web data
# - Combined answers in /get

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

# -----------------------
# FLASK APP INIT
# -----------------------
app = Flask(
    __name__,
    template_folder="templates",   # chat.html
    static_folder="src/static"     # style.css
)

# -----------------------
# ENV + KEYS
# -----------------------
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

# -----------------------
# EMBEDDINGS (HuggingFace)
# -----------------------
embeddings = download_hugging_face_embeddings()

# -----------------------
# PDF RAG via PINECONE
# -----------------------
from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
pdf_index_name = "finance-chatbot"   # your existing Pinecone index for PDFs

pdf_docsearch = PineconeVectorStore.from_existing_index(
    index_name=pdf_index_name,
    embedding=embeddings
)
pdf_retriever = pdf_docsearch.as_retriever(search_kwargs={"k": 3})

# -----------------------
# CHROMA for CSV + WEB
# -----------------------
CHROMA_DIR = "chroma_financial"
chroma_store = None  # will be loaded/created lazily


def get_chroma_store():
    """
    Lazy-load or create global Chroma store.
    """
    global chroma_store
    if chroma_store is None:
        # If directory exists, load existing index
        if os.path.exists(CHROMA_DIR):
            chroma_store = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
    return chroma_store


def add_docs_to_chroma(docs):
    """
    Add documents to Chroma vector store (CSV + crawled web data).
    """
    global chroma_store
    if chroma_store is None:
        # Create new Chroma DB
        chroma_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
    else:
        chroma_store.add_documents(docs)
    chroma_store.persist()
    print(f"[Chroma] Indexed {len(docs)} documents. Persisted at {CHROMA_DIR}.")


# -----------------------
# LLM + PROMPT + MEMORY
# -----------------------
llm = ChatOpenAI(model="gpt-4o")

# We keep memory simple (used to log history – can be extended later)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Prompt: system uses your financial assistant text; human provides question + context
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("system", "Conversation history:\n{chat_history}"),
        ("human",
            "Question: {input}\n\n"
            "Relevant context:\n{context}\n\n"
            "Answer clearly and concisely."
        ),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)

# -----------------------
# SMALL HELPERS
# -----------------------

USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}


def df_to_doc(df: pd.DataFrame, url: str, table_label: str) -> Document:
    """
    Convert a pandas DataFrame into a Document suitable for RAG.
    """
    # Convert table to CSV-like text (easier for the model to read)
    table_text = df.to_csv(index=False)
    content = (
        f"Table: {table_label}\n"
        f"Source URL: {url}\n\n"
        f"{table_text}"
    )
    return Document(
        page_content=content,
        metadata={"source": url, "table_label": table_label}
    )


def normalize_money(value: str):
    """
    Convert strings like '$325.4B', '1,234.5M', '307B', '12.3K' into numeric floats.
    If conversion fails, return the original string.
    """
    if not isinstance(value, str):
        return value
    v = value.strip().replace("$", "").replace(",", "").upper()
    multiplier = 1.0
    if v.endswith("B"):
        multiplier = 1e9
        v = v[:-1]
    elif v.endswith("M"):
        multiplier = 1e6
        v = v[:-1]
    elif v.endswith("K"):
        multiplier = 1e3
        v = v[:-1]

    try:
        num = float(v)
        return num * multiplier
    except ValueError:
        return value


def clean_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to normalize monetary / numeric columns in a DataFrame.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        # Try to normalize each column if it looks like numeric/money
        df_clean[col] = df_clean[col].apply(normalize_money)
    return df_clean


# -----------------------
# CRAWLERS
# -----------------------

def crawl_macrotrends(url: str):
    """
    Crawl a Macrotrends URL and return (docs, summary_text).
    Example: https://www.macrotrends.net/stocks/charts/GOOG/alphabet/revenue
    """
    print("[Macrotrends] Crawling:", url)
    resp = requests.get(url, headers=USER_AGENT, timeout=30)
    if resp.status_code != 200:
        return [], f"Macrotrends HTTP {resp.status_code} for {url}"

    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = soup.find("h1")
    heading = h1.get_text(strip=True) if h1 else ""

    # Get all tables; most financial tables on Macrotrends have class "historical_data_table"
    tables = soup.find_all("table")
    docs = []

    # Add a small text document for general page info
    page_info_text = f"Macrotrends page title: {title}\nHeading: {heading}\nURL: {url}"
    docs.append(
        Document(
            page_content=page_info_text,
            metadata={"source": url, "type": "page_info", "site": "macrotrends"}
        )
    )

    table_count = 0
    for idx, table in enumerate(tables):
        try:
            dfs = pd.read_html(str(table))
        except Exception as e:
            print(f"[Macrotrends] read_html error on table {idx}: {e}")
            continue

        if not dfs:
            continue

        df = dfs[0]
        # Skip tiny or meaningless tables
        if df.shape[0] < 2 or df.shape[1] < 2:
            continue

        df = clean_numeric_df(df)
        table_count += 1
        label = f"macrotrends_table_{table_count}"
        docs.append(df_to_doc(df, url, label))

    summary = (
        f"Macrotrends data extracted from '{heading or title or 'unknown'}' "
        f"at {url}. Total tables parsed: {table_count}."
    )
    return docs, summary


def crawl_stockanalysis(url: str):
    """
    Crawl a StockAnalysis.com financial page.
    Example: https://stockanalysis.com/stocks/goog/financials/
    """
    print("[StockAnalysis] Crawling:", url)
    resp = requests.get(url, headers=USER_AGENT, timeout=30)
    if resp.status_code != 200:
        return [], f"StockAnalysis HTTP {resp.status_code} for {url}"

    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = soup.find("h1")
    heading = h1.get_text(strip=True) if h1 else ""

    # StockAnalysis uses regular <table> elements for financial statements
    tables = soup.find_all("table")
    docs = []

    page_info_text = f"StockAnalysis page title: {title}\nHeading: {heading}\nURL: {url}"
    docs.append(
        Document(
            page_content=page_info_text,
            metadata={"source": url, "type": "page_info", "site": "stockanalysis"}
        )
    )

    table_count = 0
    for idx, table in enumerate(tables):
        try:
            dfs = pd.read_html(str(table))
        except Exception as e:
            print(f"[StockAnalysis] read_html error on table {idx}: {e}")
            continue

        if not dfs:
            continue

        df = dfs[0]
        if df.shape[0] < 2 or df.shape[1] < 2:
            continue

        df = clean_numeric_df(df)
        table_count += 1
        label = f"stockanalysis_table_{table_count}"
        docs.append(df_to_doc(df, url, label))

    summary = (
        f"StockAnalysis data extracted from '{heading or title or 'unknown'}' "
        f"at {url}. Total tables parsed: {table_count}."
    )
    return docs, summary


def crawl_companiesmarketcap(url: str):
    """
    Crawl a CompaniesMarketCap.com page.
    Example: https://companiesmarketcap.com/alphabet/market-cap/
    """
    print("[CompaniesMarketCap] Crawling:", url)
    resp = requests.get(url, headers=USER_AGENT, timeout=30)
    if resp.status_code != 200:
        return [], f"CompaniesMarketCap HTTP {resp.status_code} for {url}"

    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = soup.find("h1")
    heading = h1.get_text(strip=True) if h1 else ""

    tables = soup.find_all("table")
    docs = []

    page_info_text = f"CompaniesMarketCap page title: {title}\nHeading: {heading}\nURL: {url}"
    docs.append(
        Document(
            page_content=page_info_text,
            metadata={"source": url, "type": "page_info", "site": "companiesmarketcap"}
        )
    )

    table_count = 0
    for idx, table in enumerate(tables):
        try:
            dfs = pd.read_html(str(table))
        except Exception as e:
            print(f"[CompaniesMarketCap] read_html error on table {idx}: {e}")
            continue

        if not dfs:
            continue

        df = dfs[0]
        if df.shape[0] < 2 or df.shape[1] < 2:
            continue

        df = clean_numeric_df(df)
        table_count += 1
        label = f"companiesmarketcap_table_{table_count}"
        docs.append(df_to_doc(df, url, label))

    summary = (
        f"CompaniesMarketCap data extracted from '{heading or title or 'unknown'}' "
        f"at {url}. Total tables parsed: {table_count}."
    )
    return docs, summary


def crawl_financial_url(url: str):
    """
    Detect which site the URL belongs to and call the right crawler.
    Supports:
    - Macrotrends
    - StockAnalysis
    - CompaniesMarketCap
    """
    if "macrotrends.net" in url:
        return crawl_macrotrends(url)
    elif "stockanalysis.com" in url:
        return crawl_stockanalysis(url)
    elif "companiesmarketcap.com" in url:
        return crawl_companiesmarketcap(url)
    else:
        return [], "Unsupported financial site. Use Macrotrends, StockAnalysis, or CompaniesMarketCap URLs."


# -----------------------
# ROUTES
# -----------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/ingest_url", methods=["POST"])
def ingest_url_route():
    """
    Ingest a financial URL (Macrotrends / StockAnalysis / CompaniesMarketCap)
    and index its tables into Chroma.
    """
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify({"status": "error", "message": "URL is required."}), 400

    print("[Ingest URL] Received:", url)
    docs, summary = crawl_financial_url(url)

    if not docs:
        return jsonify({"status": "error", "message": summary}), 400

    add_docs_to_chroma(docs)

    return jsonify({
        "status": "ok",
        "message": f"URL {url} crawled and indexed successfully.",
        "summary": summary,
        "docs_indexed": len(docs)
    })


@app.route("/get", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    Combines:
    - PDF RAG (Pinecone)
    - Web/CSV RAG (Chroma, if available)
    """
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a question.", 200

    print("User input:", msg)

    # --- 1) Collect relevant docs from PDFs (Pinecone) ---
    combined_docs = []
    try:
        pdf_docs = pdf_retriever.get_relevant_documents(msg)
        combined_docs.extend(pdf_docs)
        print(f"[RAG] Retrieved {len(pdf_docs)} PDF docs.")
    except Exception as e:
        print("[RAG] PDF retriever error:", e)

    # --- 2) Collect relevant docs from Chroma (web/CSV) ---
    chroma = get_chroma_store()
    if chroma is not None:
        try:
            web_retriever = chroma.as_retriever(search_kwargs={"k": 4})
            web_docs = web_retriever.get_relevant_documents(msg)
            combined_docs.extend(web_docs)
            print(f"[RAG] Retrieved {len(web_docs)} Chroma docs.")
        except Exception as e:
            print("[RAG] Chroma retriever error:", e)
    else:
        print("[RAG] No Chroma store yet (no CSV or URL ingested).")

    # --- 3) If we have any docs, ask LLM with context ---
    if combined_docs:
        response = question_answer_chain.invoke(
    {
        "input": msg,
        "context": combined_docs,
        "chat_history": memory.load_memory_variables({}).get("chat_history", "")
    }
)
        # create_stuff_documents_chain usually returns {"output_text": "..."}
        answer = response.get("output_text") if isinstance(response, dict) else str(response)
    else:
        answer = (
            "I couldn't find any relevant information yet.\n\n"
            "Please make sure you have:\n"
            "- Uploaded some PDFs (already indexed in Pinecone), or\n"
            "- Crawled a financial URL via Macrotrends / StockAnalysis / CompaniesMarketCap."
        )

    # Save to memory (not heavily used yet, but ready for future)
    memory.save_context({"input": msg}, {"output": answer})

    print("Response:", answer)
    return str(answer)


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
