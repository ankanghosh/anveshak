import os
import json
import numpy as np
import faiss
import torch
import torch.nn as nn
from google.cloud import storage
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import unicodedata
import streamlit as st
from utils import setup_gcp_auth, setup_openai_auth
import gc

# Force model to CPU for stability and to avoid GPU memory issues on resource-constrained environments
# This is especially important for deployment on platforms like Hugging Face Spaces
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define local paths for files downloaded from Google Cloud Storage
# These files are cached locally to avoid repeated downloads and improve performance
local_embeddings_file = "all_embeddings.npy"
local_faiss_index_file = "faiss_index.faiss"
local_text_chunks_file = "text_chunks.txt"
local_metadata_file = "metadata.jsonl"

# =============================================================================
# RESOURCE CACHING
# =============================================================================

@st.cache_resource(show_spinner=False)
def cached_load_model():
    """
    Load and cache the E5-large-v2 embedding model and tokenizer.
    
    Uses Streamlit's cache_resource decorator to ensure the model
    is loaded only once during the application session, improving 
    performance and reducing memory usage.
    
    Returns:
        tuple: (tokenizer, model) pair or (None, None) if loading fails
    """
    try:
        # Force model to CPU for stability
        device = torch.device("cpu")
        
        # Get embedding model path from secrets
        try:
            embedding_model = st.secrets["EMBEDDING_MODEL"]
        except KeyError:
            print("❌ Error: Embedding model path not found in secrets")
            return None, None
            
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        model = AutoModel.from_pretrained(
            embedding_model,
            torch_dtype=torch.bfloat16
        )
        
        # Move model to CPU and set to eval mode for inference
        model = model.to(device)
        model.eval()
        
        # Disable gradient computation to save memory during inference
        torch.set_grad_enabled(False)
        
        print("✅ Model loaded successfully (cached)")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

@st.cache_resource(show_spinner=False)
def cached_load_data_files():
    """
    Load and cache data files needed for the RAG system.
    
    This function loads:
    - FAISS index for vector similarity search
    - Text chunks containing the original spiritual text passages
    - Metadata dictionary with publication and author information
    - OpenAI client for answer generation
    
    All files are downloaded from Google Cloud Storage if not already present locally.
    
    Returns:
        tuple: (faiss_index, text_chunks, metadata_dict, openai_client) or (None, None, None, None) if loading fails
    """
    # Initialize GCP and OpenAI clients
    bucket = setup_gcp_client()
    openai_client = setup_openai_client()
    
    if not bucket or not openai_client:
        print("Failed to initialize required services")
        return None, None, None, None
    
    # Get GCS paths from secrets - required
    try:
        metadata_file_gcs = st.secrets["METADATA_PATH_GCS"]
        embeddings_file_gcs = st.secrets["EMBEDDINGS_PATH_GCS"]
        faiss_index_file_gcs = st.secrets["INDICES_PATH_GCS"]
        text_chunks_file_gcs = st.secrets["CHUNKS_PATH_GCS"]
    except KeyError as e:
        print(f"❌ Error: Required GCS path not found in secrets: {e}")
        return None, None, None, None
    
    # Download necessary files if not already present locally
    success = True
    success &= download_file_from_gcs(bucket, faiss_index_file_gcs, local_faiss_index_file)
    success &= download_file_from_gcs(bucket, text_chunks_file_gcs, local_text_chunks_file)
    success &= download_file_from_gcs(bucket, metadata_file_gcs, local_metadata_file)
    
    if not success:
        print("Failed to download required files")
        return None, None, None, None
    
    # Load FAISS index
    try:
        faiss_index = faiss.read_index(local_faiss_index_file)
    except Exception as e:
        print(f"❌ Error loading FAISS index: {str(e)}")
        return None, None, None, None
    
    # Load text chunks
    try:
        text_chunks = {}  # Mapping: ID -> (Title, Author, Text)
        with open(local_text_chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    text_chunks[int(parts[0])] = (parts[1], parts[2], parts[3])
    except Exception as e:
        print(f"❌ Error loading text chunks: {str(e)}")
        return None, None, None, None
    
    # Load metadata
    try:
        metadata_dict = {}
        with open(local_metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                metadata_dict[item["Title"]] = item
    except Exception as e:
        print(f"❌ Error loading metadata: {str(e)}")
        return None, None, None, None
    
    print(f"✅ Data loaded successfully (cached): {len(text_chunks)} passages available")
    return faiss_index, text_chunks, metadata_dict, openai_client

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_gcp_client():
    """
    Initialize and return the Google Cloud Storage client.
    
    Sets up GCP authentication and creates a client for the configured bucket.
    
    Returns:
        google.cloud.storage.bucket.Bucket: The GCS bucket object or None if initialization fails
    """
    try:
        credentials = setup_gcp_auth()
        try:
            bucket_name_gcs = st.secrets["BUCKET_NAME_GCS"]
        except KeyError:
            print("❌ Error: GCS bucket name not found in secrets")
            return None
            
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(bucket_name_gcs)
        print("✅ GCP client initialized successfully")
        return bucket
    except Exception as e:
        print(f"❌ GCP client initialization error: {str(e)}")
        return None

def setup_openai_client():
    """
    Initialize the OpenAI client.
    
    Sets up OpenAI API authentication for generating answers using the LLM.
    
    Returns:
        OpenAI: Configured OpenAI client instance, or None if initialization fails
    """
    try:
        client = setup_openai_auth()
        print("✅ OpenAI client initialized successfully")
        return client
    except Exception as e:
        print(f"❌ OpenAI client initialization error: {str(e)}")
        return None

def download_file_from_gcs(bucket, gcs_path, local_path):
    """
    Download a file from Google Cloud Storage to local storage.
    
    Only downloads if the file doesn't already exist locally to avoid
    redundant network transfers and improve startup time.
    
    Args:
        bucket: GCS bucket object
        gcs_path (str): Path to file in GCS bucket
        local_path (str): Local path where file should be saved
        
    Returns:
        bool: True if download successful or file already exists, False otherwise
    """
    try:
        if os.path.exists(local_path):
            print(f"âœ… File already exists locally: {local_path}")
            return True
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"âœ… Downloaded {gcs_path} → {local_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {gcs_path}: {str(e)}")
        return False

def get_embedding(text, tokenizer, model):
    """
    Generate a vector embedding for a given text using the E5-large-v2 model.
    
    This function:
    1. Adds the required "query:" prefix for E5 model
    2. Tokenizes the text
    3. Runs it through the model
    4. Performs mean pooling on the output
    5. Normalizes the resulting vector
    
    Args:
        text (str): Input text to embed
        tokenizer: Hugging Face tokenizer
        model: Hugging Face model
        
    Returns:
        numpy.ndarray: Normalized embedding vector of shape (1024,) or None if error occurs
    """
    try:
        device = torch.device("cpu")
        prefixed_text = f"query: {text}"
        inputs = tokenizer(prefixed_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).cpu().numpy()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        # Clean up
        del inputs, outputs, token_embeddings, input_mask_expanded, sum_embeddings, sum_mask
        gc.collect()
        
        return embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {str(e)}")
        return None

# =============================================================================
# RAG PIPELINE FUNCTIONS
# =============================================================================

def retrieve_passages(query, faiss_index, text_chunks, metadata_dict, top_k=5):
    """
    Retrieve the most relevant text passages for a given query.
    
    This function:
    1. Generates an embedding for the user's query
    2. Searches the FAISS index for similar passages
    3. Deduplicates results to ensure variety of sources
    4. Returns passages with their metadata
    
    Args:
        query (str): The user's spiritual question
        faiss_index: FAISS index containing passage embeddings
        text_chunks (dict): Dictionary mapping IDs to (title, author, text) tuples
        metadata_dict (dict): Dictionary mapping titles to full metadata
        top_k (int): Number of unique sources to retrieve
        
    Returns:
        tuple: (list of passage texts, list of (title, author, publisher) tuples)
    """
    try:
        print(f"🔍 Retrieving passages for query: {query}")
        tokenizer, model = cached_load_model()
        if tokenizer is None or model is None:
            print("❌ Model not available")
            return [], []
        
        query_embedding = get_embedding(query, tokenizer, model)
        if query_embedding is None:
            print("❌ Failed to generate query embedding")
            return [], []
        
        # Search FAISS index
        # Request more results initially to allow for deduplication
        search_k = min(top_k * 3, faiss_index.ntotal)
        distances, indices = faiss_index.search(query_embedding.astype(np.float32), search_k)
        
        print(f"Found {len(indices[0])} potential matches")
        for i, (dist, idx) in enumerate(zip(distances[0][:10], indices[0][:10])):
            print(f"Distance: {dist:.4f}, Index: {idx}")
        
        # Deduplicate by title
        retrieved_passages = []
        retrieved_sources = []
        cited_titles = set()
        
        for idx in indices[0]:
            if idx < len(text_chunks):
                title_with_txt, author, text = text_chunks[idx]
                clean_title = title_with_txt.replace(".txt", "") if title_with_txt.endswith(".txt") else title_with_txt
                clean_title = unicodedata.normalize("NFC", clean_title)
                if clean_title in cited_titles:
                    continue  
                metadata_entry = metadata_dict.get(clean_title, {})
                author = metadata_entry.get("Author", "Unknown")
                publisher = metadata_entry.get("Publisher", "Unknown")
                cited_titles.add(clean_title)
                retrieved_passages.append(text)
                retrieved_sources.append((clean_title, author, publisher))
                if len(retrieved_passages) == top_k:
                    break
        print(f"Retrieved {len(retrieved_passages)} passages")
        return retrieved_passages, retrieved_sources
    except Exception as e:
        print(f"❌ Error in retrieve_passages: {str(e)}")
        return [], []

def answer_with_llm(query, openai_client, context=None, word_limit=200):
    """
    Generate an answer using the OpenAI GPT model with formatted citations.
    
    This function:
    1. Formats retrieved passages with source information
    2. Creates a prompt with system and user messages
    3. Calls the OpenAI API to generate an answer
    4. Trims the response to the specified word limit
    
    The system prompt ensures answers maintain appropriate respect for spiritual traditions,
    synthesize rather than quote directly, and acknowledge gaps when relevant information
    isn't available.
    
    Args:
        query (str): The user's spiritual question
        openai_client (OpenAI): Configured OpenAI client instance
        context (list, optional): List of (source_info, text) tuples for context
        word_limit (int): Maximum word count for the generated answer
        
    Returns:
        str: The generated answer or an error message
    """
    try:
        if context:
            formatted_contexts = []
            total_chars = 0
            max_context_chars = 4000  # Limit context size to avoid exceeding token limits
            for (title, author, publisher), text in context:
                remaining_space = max(0, max_context_chars - total_chars)
                excerpt_len = min(150, remaining_space)
                if excerpt_len > 50:
                    excerpt = text[:excerpt_len].strip() + "..." if len(text) > excerpt_len else text
                    formatted_context = f"[{title} by {author}, Published by {publisher}] {excerpt}"
                    formatted_contexts.append(formatted_context)
                    total_chars += len(formatted_context)
                if total_chars >= max_context_chars:
                    break
            formatted_context = "\n".join(formatted_contexts)
        else:
            formatted_context = "No relevant information available."

        system_message = (
            "You are an AI specialized in spirituality, primarily based on Indian spiritual texts and teachings."
            "While your knowledge is predominantly from Indian spiritual traditions, you also have limited familiarity with spiritual concepts from other global traditions."
            "Answer based on context, summarizing ideas rather than quoting verbatim."
            "If no relevant information is found in the provided context, politely inform the user that this specific query may not be covered in the available spiritual texts. Suggest they try a related question or rephrase their query or try a different query."
            "Avoid repetition and irrelevant details."
            "Ensure proper citation and do not include direct excerpts."
            "Maintain appropriate, respectful language at all times."
            "Do not use profanity, expletives, obscenities, slurs, hate speech, sexually explicit content, or language promoting violence."
            "As a spiritual guidance system, ensure all responses reflect dignity, peace, love, and compassion consistent with spiritual traditions."
            "Provide concise, focused answers without lists or lengthy explanations."
        )

        user_message = f"""
        Context:
        {formatted_context}
        Question:
        {query}
        """

        try:
            llm_model = st.secrets["LLM_MODEL"]
        except KeyError:
            print("❌ Error: LLM model not found in secrets")
            return "I apologize, but I am unable to answer at the moment."
            
        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        # Extract the answer and apply word limit
        answer = response.choices[0].message.content.strip()
        words = answer.split()
        if len(words) > word_limit:
            answer = " ".join(words[:word_limit])
            if not answer.endswith((".", "!", "?")):
                answer += "."
        return answer
    except Exception as e:
        print(f"❌ LLM API error: {str(e)}")
        return "I apologize, but I am unable to answer at the moment."

def format_citations(sources):
    """
    Format citations for display to the user.
    
    Creates properly formatted citations for each source used in generating the answer.
    Each citation appears on a new line with consistent formatting.
    
    Args:
        sources (list): List of (title, author, publisher) tuples
        
    Returns:
        str: Formatted citations as a string with each citation on a new line
    """
    formatted_citations = []
    for title, author, publisher in sources:
        if publisher.endswith(('.')):
            formatted_citations.append(f"📚 {title} by {author}, Published by {publisher}")
        else:
            formatted_citations.append(f"📚 {title} by {author}, Published by {publisher}.")
    return "\n".join(formatted_citations)

# =============================================================================
# DATA CACHING FOR QUERY RESULTS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def cached_process_query(query, top_k=5, word_limit=200):
    """
    Process a user query with caching to avoid redundant computation.
    
    This function is cached with a Time-To-Live (TTL) of 1 hour, meaning identical 
    queries within this time period will return cached results rather than 
    reprocessing, improving responsiveness.
    
    Args:
        query (str): The user's spiritual question
        top_k (int): Number of sources to retrieve and use for answer generation
        word_limit (int): Maximum word count for the generated answer
        
    Returns:
        dict: Dictionary containing the query, answer, and citations
    """
    print(f"\n🔍 Processing query (cached): {query}")
    # Load all necessary data resources (with caching)
    faiss_index, text_chunks, metadata_dict, openai_client = cached_load_data_files()
    # Handle missing data gracefully
    if faiss_index is None or text_chunks is None or metadata_dict is None or openai_client is None:
        return {
            "query": query, 
            "answer_with_rag": "⚠️ System error: Data files not loaded properly.", 
            "citations": "No citations available."
        }
    # Step 1: Retrieve relevant passages using similarity search    
    retrieved_context, retrieved_sources = retrieve_passages(
        query, 
        faiss_index, 
        text_chunks, 
        metadata_dict,
        top_k=top_k
    )
    # Step 2: Format citations for display
    sources = format_citations(retrieved_sources) if retrieved_sources else "No citation available."
    # Step 3: Generate the answer if relevant context was found
    if retrieved_context:
        context_with_sources = list(zip(retrieved_sources, retrieved_context))
        llm_answer_with_rag = answer_with_llm(query, openai_client, context_with_sources, word_limit=word_limit)
    else:
        llm_answer_with_rag = "⚠️ No relevant context found."
    # Return the complete response package
    return {"query": query, "answer_with_rag": llm_answer_with_rag, "citations": sources}

def process_query(query, top_k=5, word_limit=200):
    """
    Process a query through the RAG pipeline with proper formatting.
    
    This is the main entry point for query processing, wrapping the cached 
    query processing function.
    
    Args:
        query (str): The user's spiritual question
        top_k (int): Number of sources to retrieve and use for answer generation
        word_limit (int): Maximum word count for the generated answer
        
    Returns:
        dict: Dictionary containing the query, answer, and citations
    """
    return cached_process_query(query, top_k, word_limit)

# Alias for backward compatibility
load_model = cached_load_model