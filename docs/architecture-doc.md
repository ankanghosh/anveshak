# Architecture Document

This document provides a detailed overview of the architecture, component interactions, and technical design decisions encompassing Anveshak: Spirituality Q&A.

## System Architecture Overview

Anveshak: Spirituality Q&A follows a Retrieval-Augmented Generation (RAG) architecture pattern, combining information retrieval with language generation to produce factual, grounded answers to spiritual questions.

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONT-END LAYER                            │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  Main App Page  │  │  Sources Page   │  │  Publishers Page    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│                                                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           BACKEND LAYER                             │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Query Processor │  │ Retrieval Engine│  │ Generation Engine   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│                                                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   FAISS Index   │  │  Text Chunks    │  │     Metadata        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Front-end Layer

The front-end layer is built with Streamlit and consists of multiple pages:

#### Main App Page (`app.py`)
- Provides the question input interface
- Displays answers and citations
- Offers configurable parameters (number of sources, word limit)
- Shows pre-selected common spiritual questions
- Contains information about the application and disclaimers
- Contains acknowledgment sections

#### Sources Page (`1_Sources.py`)
- Lists all spiritual texts and traditions used in Anveshak: Spirituality Q&A
- Provides information about the Saints and Spiritual Masters
- Organizes sources by tradition and category

#### Publishers Page (`2_Publishers.py`)
- Acknowledges all publishers whose works are referenced
- Explains copyright considerations and fair use

#### Contacts Page (`3_Contacts.py`)
- Provides contact information for feedback and questions
- Explains the purpose and limitations of Anveshak: Spirituality Q&A

### 2. Backend Layer

The backend layer handles the core functionality of processing queries, retrieving relevant passages, and generating answers.

#### Query Processor
- Takes user queries from the front-end
- Manages the end-to-end processing flow
- Caches results to improve performance
- Formats and returns answers with citations

```python
@st.cache_data(ttl=3600, show_spinner=False)
def cached_process_query(query, top_k=5, word_limit=100):
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

def process_query(query, top_k=5, word_limit=100):
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
```

#### Retrieval Engine
- Generates embeddings for user queries
- Performs similarity search in the FAISS index
- Retrieves the most relevant text chunks
- Adds metadata to the retrieved passages

```python
ddef retrieve_passages(query, faiss_index, text_chunks, metadata_dict, top_k=5, similarity_threshold=0.5):
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
```

#### Generation Engine
- Takes retrieved passages as context
- Uses OpenAI's GPT model to generate answers
- Ensures answers respect the word limit
- Formats the output with proper citations

```python
def answer_with_llm(query, context=None, word_limit=100):
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

### 3. Data Layer

The data layer stores and manages the embedded text chunks, metadata, and vector indices:

#### FAISS Index
- Stores vector embeddings of all text chunks
- Enables efficient similarity search with cosine similarity
- Provides fast retrieval for Anveshak

```python
# Building the FAISS index (during preprocessing)
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity for normalized vectors)
index.add(all_embeddings)
```

#### Text Chunks
- Contains the actual text content split into manageable chunks
- Stores text with unique identifiers that map to the FAISS index
- Formatted as tab-separated values with IDs, titles, authors, and content

```
# Format of text_chunks.txt
ID  Title  Author  Text_Content
0   Bhagavad Gita   Vyasa   The supreme Lord said: I have taught this imperishable yoga to Vivasvan...
1   Yoga Sutras Patanjali   Yogas chitta vritti nirodhah - Yoga is the stilling of the fluctuations...
...
```

#### Metadata
- Stores additional information about each source text
- Includes author information, publisher details, copyright information, and more
- Used to provide accurate citations for answers

```json
// Example metadata.jsonl entry
{"Title": "Example Title", "Author": "Example Author", "Publisher": "Example Publisher", "URL": "https://example.com", "Uploaded": true}
```

## Data Flow and Processing

### 1. Preprocessing Pipeline

The preprocessing pipeline runs offline to prepare the text corpus:

```
Raw Texts → Cleaning → Chunking → Embedding → Indexing → GCS Storage
```

Each step is handled by specific functions in the `preprocessing.py` script:

1. **Text Collection**: Texts are collected from various sources and uploaded to Google Cloud Storage
2. **Text Cleaning**: HTML and formatting artifacts are removed using `rigorous_clean_text()`
3. **Text Chunking**: Long texts are split into manageable chunks with `chunk_text()`
4. **Embedding Generation**: Text chunks are converted to vector embeddings using `create_embeddings()`
5. **Index Building**: Embeddings are added to a FAISS index for efficient retrieval
6. **Storage**: All processed data is stored in Google Cloud Storage for Anveshak to access

### 2. Query Processing Flow

When a user submits a question, the system follows this flow:

1. **Query Embedding**: The user's question is embedded using the same model as the text corpus
2. **Similarity Search**: The query embedding is compared against the FAISS index to find similar text chunks
3. **Context Assembly**: Retrieved chunks are combined with their metadata to form the context
4. **Answer Generation**: The context and query are sent to the Large Language Model (LLM) to generate an answer
5. **Citation Formatting**: Sources are formatted as citations to accompany the answer
6. **Result Presentation**: The answer and citations are displayed to the user

## Caching Strategy

Anveshak implements a multi-level caching strategy to optimize performance:

### Resource Caching
- Model and data files are cached using `@st.cache_resource`
- Ensures the embedding model and FAISS index are loaded only once during the session

```python
@st.cache_resource(show_spinner=False)
def cached_load_model():
    # Load embedding model once and cache it
    
@st.cache_resource(show_spinner=False)
def cached_load_data_files():
    # Load FAISS index, text chunks, and metadata once and cache them
```

### Data Caching
- Query results are cached using `@st.cache_data` with a Time-To-Live (TTL) of 1 hour
- Prevents redundant processing of identical queries

```python
@st.cache_data(ttl=3600, show_spinner=False)
def cached_process_query(query, top_k=5, word_limit=100):
    # Cache query results for an hour
```

### Session State Management
- Streamlit session state is used to manage UI state and user interactions
- Prevents unnecessary recomputation during re-renders

```python
...
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
...    
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
# ... and more session state variables
```

## Authentication and Security

Anveshak uses two authentication systems:

### Google Cloud Storage Authentication
- Authenticates with GCS to access stored data
- Uses service account credentials stored securely

```python
Anveshak: Spirituality Q&A uses two authentication systems:

### Google Cloud Storage Authentication
- Authenticates with GCS to access stored data
- Uses service account credentials stored exclusively in Hugging Face Spaces secrets for production deployment
- Supports alternative authentication methods (environment variables, Streamlit secrets) for development environments

```python
def setup_gcp_auth():
    """Setup Google Cloud Platform (GCP) authentication using various methods.
    
    This function tries multiple authentication methods in order of preference:
    1. HF Spaces environment variable (GCP_CREDENTIALS) - primary production method
    2. Local environment variable pointing to credentials file (GOOGLE_APPLICATION_CREDENTIALS)
    3. Streamlit secrets (gcp_credentials)
    
    Note: In production, credentials are stored exclusively in HF Spaces secrets.
    """
    # Try multiple authentication methods and return credentials.```

### OpenAI API Authentication
- Authenticates with OpenAI to use their LLM API
- Uses API key stored securely

```python
def setup_openai_auth():
    """Setup OpenAI API authentication using various methods.
    
    This function tries multiple authentication methods in order of preference:
    1. Standard environment variable (OPENAI_API_KEY)
    2. HF Spaces environment variable (OPENAI_KEY) - primary production method
    3. Streamlit secrets (openai_api_key)
    
    Note: In production, the API key is stored exclusively in HF Spaces secrets.
    """
    # Try multiple authentication methods to set up the API key
```

## Privacy Considerations

Anveshak: Spirituality Q&A is designed with privacy in mind:

1. **No Data Collection**: The application does not save user data or queries
2. **Stateless Operation**: Each query is processed independently
3. **No User Tracking**: No analytics or tracking mechanisms are implemented
4. **Local Processing**: Embedding generation happens locally when possible

## Deployment Architecture

Anveshak: Spirituality Q&A is deployed on Hugging Face Spaces, which provides:

- Containerized environment
- Git-based deployment
- Secret management for API keys and credentials
- Persistent storage for cached files
- Continuous availability

The deployment process involves:
1. Pushing code to GitHub
2. Connecting the GitHub repository to Hugging Face Spaces
3. Configuring environment variables and secrets in the Hugging Face UI
4. Automatic deployment when changes are pushed to the repository

## Technical Design Decisions

### Choice of Embedding Model
- **Selected Model**: E5-large-v2
- **Justification**: 
  - Strong performance on information retrieval tasks
  - Good balance between accuracy and computational efficiency
  - Supports semantic understanding of spiritual concepts

### Vector Search Implementation
- **Selected Technology**: FAISS with IndexFlatIP
- **Justification**:
  - Optimized for inner product (cosine similarity) search
  - Exact search rather than approximate for maximum accuracy
  - Small enough index to fit in memory for this application

### LLM Selection
- **Selected Model**: OpenAI GPT-3.5 Turbo
- **Justification**:
  - Powerful context understanding
  - Strong ability to synthesize information from multiple sources
  - Good balance between accuracy and cost

### Front-end Framework
- **Selected Technology**: Streamlit
- **Justification**:
  - Rapid development of data-focused applications
  - Built-in caching mechanisms
  - Easy deployment on Hugging Face Spaces
  - Simple, intuitive UI for non-technical users

### Response Format
- **Design Choice**: Concise, direct answers
- **Justification**:
  - Spiritual wisdom often benefits from simplicity and directness
  - Avoids overwhelming users with excessive information
  - Maintains focus on the core of the question

## Limitations and Constraints

1. **Context Window Limitations**: The LLM has a maximum context window, limiting the amount of text that can be included in each query.
   - Mitigation: Text chunks are limited to 500 words, and only a subset of the most relevant chunks are included in the context.

2. **Embedding Model Accuracy**: No embedding model perfectly captures the semantics of spiritual texts.
   - Mitigation: Use of a high-quality embedding model (E5-large-v2) and a similarity threshold to filter out less relevant results.

3. **Resource Constraints**: Hugging Face Spaces has limited computational resources.
   - Mitigation: Forcing CPU usage for the embedding model, implementing aggressive caching, and optimizing memory usage.

4. **Copyright Considerations**: Anveshak: Spirituality Q&A respects copyright while providing valuable information.
   - Implementation: Word limits on responses, proper citations for all sources, and encouragement for users to purchase original texts.

5. **Language Limitations**: Currently, Anveshak is only available in English.
   - Mitigation: Future plans include support for multiple Indian languages.

## Future Architecture Extensions

1. **Multi-language Support**: Add capability to process and answer questions in Sanskrit, Hindi, Bengali, Tamil, and other Indian languages.

2. **Hybrid Retrieval**: Implement a combination of dense and sparse retrieval to improve passage selection.

3. **Local LLM Integration**: Use a self-hosted open-source alternative for the LLM.

4. **User Feedback Loop**: Add a mechanism for users to rate answers and use this feedback to improve retrieval.

5. **Advanced Caching**: Implement a distributed caching system for better performance at scale.

## Conclusion

The architecture of Anveshak balances technical sophistication with simplicity and accessibility. By combining modern NLP techniques with traditional spiritual texts, it creates a bridge between ancient wisdom and contemporary technology, making spiritual knowledge more accessible to seekers around the world.

Anveshak: Spirituality Q&A acknowledges and honors Saints, Sages, Siddhas, Yogis, Sadhus, Rishis, Gurus, Mystics, and Spiritual Masters from all backgrounds, genders, traditions, and walks of life, understanding that wisdom transcends all such distinctions. Its focused approach on providing concise, direct answers maintains the essence of spiritual teaching while embracing modern technological capabilities.