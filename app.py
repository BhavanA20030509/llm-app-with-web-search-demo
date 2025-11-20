"""
LLM App with Progressive Web Search Streaming
"""

import asyncio
import os
import tempfile
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import chromadb
import ollama
import streamlit as st
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
from duckduckgo_search import DDGS
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

# ------------------------------
# System prompt
# ------------------------------
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.
Context will be passed as "Context:" and User question as "Question:".
Answer only based on the context. If no context is provided, say you have no context.
"""

# ------------------------------
# LLM call function
# ------------------------------
def call_llm(prompt: str, context: str | None = None):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}, Question: {prompt}" if context else prompt},
    ]
    response = ollama.chat(model="llama3.2:3b", stream=True, messages=messages)
    for chunk in response:
        if not chunk.get("done"):
            yield chunk["message"]["content"]
        else:
            break

# ------------------------------
# Vector DB setup
# ------------------------------
def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db", settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_or_create_collection(
        name="web_llm",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection, chroma_client

# ------------------------------
# Helpers
# ------------------------------
def normalize_url(url):
    url = url.replace("https://", "").replace("http://", "").replace("www.", "")
    url = url.replace("/", "_").replace("-", "_").replace(".", "_")
    return url

def add_to_vector_database(results: list[CrawlResult]):
    collection, _ = get_vector_collection()
    for result in results:
        documents, metadatas, ids = [], [], []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        markdown_result = getattr(result.markdown_v2, "fit_markdown", None)
        if not markdown_result:
            continue
        temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        temp_file.write(markdown_result)
        temp_file.flush()
        loader = UnstructuredMarkdownLoader(temp_file.name, mode="single")
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        os.unlink(temp_file.name)

        normalized_url = normalize_url(result.url)
        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append({"source": result.url})
            ids.append(f"{normalized_url}_{idx}")

        if documents:
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

async def crawl_webpages(urls: list[str], prompt: str) -> list[CrawlResult]:
    bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=1.2)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0",
        page_timeout=20000,
    )
    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls, config=crawler_config)
        return results

def check_robots_txt(urls: list[str]) -> list[str]:
    allowed_urls = []
    for url in urls:
        try:
            robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
            rp = RobotFileParser(robots_url)
            rp.read()
            if rp.can_fetch("*", url):
                allowed_urls.append(url)
        except Exception:
            allowed_urls.append(url)
    return allowed_urls

def get_web_urls(search_term: str, num_results: int = 10) -> list[str]:
    discard_urls = ["youtube.com", "britannica.com", "vimeo.com"]
    for url in discard_urls:
        search_term += f" -site:{url}"
    results = DDGS().text(search_term, max_results=num_results)
    results = [result["href"] for result in results]
    return check_robots_txt(results)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="LLM Web Search")
st.header("🔍 LLM Web Search")

prompt = st.text_area("Put your query here", placeholder="Add your query...")
is_web_search = st.checkbox("Enable web search", value=False)
go = st.button("⚡️ Go")

collection, chroma_client = get_vector_collection()
placeholder = st.empty()

if prompt and go:

    # 1. Display streaming text while running
    def run_llm_with_context(prompt_text, context_text=None):
        for chunk in call_llm(prompt_text, context=context_text):
            placeholder.write(chunk)

    # 2. If web search enabled, crawl + embed asynchronously
    if is_web_search:
        try:
            web_urls = get_web_urls(prompt)
            if not web_urls:
                st.warning("No results found.")
            else:
                # Run crawl in thread to avoid blocking Streamlit
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(crawl_webpages(web_urls, prompt))
                add_to_vector_database(results)

                qresults = collection.query(query_texts=[prompt], n_results=10)
                context_docs = qresults.get("documents", [])
                context = context_docs[0] if context_docs else None

                if not context:
                    placeholder.write("No context retrieved from web.")
                else:
                    run_llm_with_context(prompt, context)

        finally:
            chroma_client.delete_collection(name="web_llm")

    else:
        run_llm_with_context(prompt)
