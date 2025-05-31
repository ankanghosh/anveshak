import os
import json
import numpy as np
import faiss
import torch
import torch.nn as nn
from google.cloud import storage
from transformers import AutoTokenizer, AutoModel
import openai
import textwrap
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
            torch_dtype=torch.float16
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
    
    All files are downloaded from Google Cloud Storage if not already present locally.
    
    Returns:
        tuple: (faiss_index, text_chunks, metadata_dict) or (None, None, None) if loading fails
    """
    # Initialize GCP and OpenAI clients
    bucket = setup_gcp_client()
    openai_initialized = setup_openai_client()
    
    if not bucket or not openai_initialized:
        print("Failed to initialize required services")
        return None, None, None
    
    # Get GCS paths from secrets - required
    try:
        metadata_file_gcs = st.secrets["METADATA_PATH_GCS"]
        embeddings_file_gcs = st.secrets["EMBEDDINGS_PATH_GCS"]
        faiss_index_file_gcs = st.secrets["INDICES_PATH_GCS"]
        text_chunks_file_gcs = st.secrets["CHUNKS_PATH_GCS"]
    except KeyError as e:
        print(f"❌ Error: Required GCS path not found in secrets: {e}")
        return None, None, None
    
    # Download necessary files if not already present locally
    success = True
    success &= download_file_from_gcs(bucket, faiss_index_file_gcs, local_faiss_index_file)
    success &= download_file_from_gcs(bucket, text_chunks_file_gcs, local_text_chunks_file)
    success &= download_file_from_gcs(bucket, metadata_file_gcs, local_metadata_file)
    
    if not success:
        print("Failed to download required files")
        return None, None, None
    
    # Load FAISS index
    try:
        faiss_index = faiss.read_index(local_faiss_index_file)
    except Exception as e:
        print(f"❌ Error loading FAISS index: {str(e)}")
        return None, None, None
    
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
        return None, None, None
    
    # Load metadata
    try:
        metadata_dict = {}
        with open(local_metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                metadata_dict[item["Title"]] = item
    except Exception as e:
        print(f"❌ Error loading metadata: {str(e)}")
        return None, None, None
    
    print(f"✅ Data loaded successfully (cached): {len(text_chunks)} passages available")
    return faiss_index, text_chunks, metadata_dict

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
        bool: True if initialization was successful, False otherwise
    """
    try:
        setup_openai_auth()
        print("✅ OpenAI client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ OpenAI client initialization error: {str(e)}")
        return False

def download_file_from_gcs(bucket, gcs_path, local_path):
    """
    Download a file from Google Cloud Storage to local storage.
    
    Only downloads if the file isn't already present locally, avoiding redundant downloads.
    
    Args:
        bucket: GCS bucket object
        gcs_path (str): Path to the file in GCS
        local_path (str): Local path where the file should be saved
        
    Returns:
        bool: True if download was successful or file already exists, False otherwise
    """
    try:
        if os.path.exists(local_path):
            print(f"File already exists locally: {local_path}")
            return True
            
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"✅ Downloaded {gcs_path} → {local_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {gcs_path}: {str(e)}")
        return False

def average_pool(last_hidden_states, attention_mask):
    """
    Perform average pooling on model outputs for sentence embeddings.
    
    This function creates a fixed-size vector representation of a text sequence by averaging
    the token embeddings, accounting for padding tokens using the attention mask.
    
    Args:
        last_hidden_states: Hidden states output from the model
        attention_mask: Attention mask indicating which tokens to include
        
    Returns:
        torch.Tensor: Pooled representation of the input sequence
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# In-memory cache for query embeddings to avoid redundant computations
query_embedding_cache = {}

def get_embedding(text):
    """
    Generate embeddings for a text query using the cached model.
    
    Uses an in-memory cache to avoid redundant embedding generation for repeated queries.
    Prefixes inputs with "query:" as required by the E5 model for search queries.
    
    Args:
        text (str): The query text to embed
        
    Returns:
        numpy.ndarray: The embedding vector or a zero vector if embedding fails
    """
    if text in query_embedding_cache:
        return query_embedding_cache[text]

    try:
        tokenizer, model = cached_load_model()
        if model is None:
            print("Model is None, returning zero embedding")
            return np.zeros((1, 1024), dtype=np.float32)
            
        # For E5 models, "query:" prefix is for questions. Passages use "passage:" prefix during preprocessing
        input_text = f"query: {text}"
        inputs = tokenizer(
            input_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            return_attention_mask=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.detach().cpu().numpy()
        del outputs, inputs
        gc.collect()
        query_embedding_cache[text] = embeddings
        return embeddings
    except Exception as e:
        print(f"❌ Embedding error: {str(e)}")
        return np.zeros((1, 1024), dtype=np.float32)

def retrieve_passages(query, faiss_index, text_chunks, metadata_dict, top_k=5, similarity_threshold=0.5):
    """
    Retrieve the most relevant passages for a given spiritual query.
    
    This function:
    1. Embeds the user query using the same model used for text chunks
    2. Finds similar passages using the FAISS index with cosine similarity
    3. Filters results based on similarity threshold to ensure relevance
    4. Enriches results with metadata (title, author, publisher)
    5. Ensures passage diversity by including only one passage per source title
    
    Args:
        query (str): The user's spiritual question
        faiss_index: FAISS index containing passage embeddings
        text_chunks (dict): Dictionary mapping IDs to text chunks and metadata
        metadata_dict (dict): Dictionary containing publication information
        top_k (int): Maximum number of passages to retrieve
        similarity_threshold (float): Minimum similarity score (0.0-1.0) for retrieved passages
        
    Returns:
        tuple: (retrieved_passages, retrieved_sources) containing the text and source information
    """
    try:
        print(f"\n🔍 Retrieving passages for query: {query}")
        query_embedding = get_embedding(query)
        distances, indices = faiss_index.search(query_embedding, top_k * 2)
        print(f"Found {len(distances[0])} potential matches")
        retrieved_passages = []
        retrieved_sources = []
        cited_titles = set()
        for dist, idx in zip(distances[0], indices[0]):
            print(f"Distance: {dist:.4f}, Index: {idx}")
            if idx in text_chunks and dist >= similarity_threshold:
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

def answer_with_llm(query, context=None, word_limit=200):
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
            
        response = openai.chat.completions.create(
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
    faiss_index, text_chunks, metadata_dict = cached_load_data_files()
    # Handle missing data gracefully
    if faiss_index is None or text_chunks is None or metadata_dict is None:
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
        llm_answer_with_rag = answer_with_llm(query, context_with_sources, word_limit=word_limit)
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