# Data Handling Explanation

This document explains how data is processed, stored, and handled in Anveshak: Spirituality Q&A, with special attention to ethical considerations and copyright respect.

## Data Sources

### Text Corpus Overview

Anveshak: Spirituality Q&A uses approximately 133 digitized spiritual texts sourced from freely available resources. These texts include:

- Ancient sacred literature (Vedas, Upanishads, Puranas, Sutras, Dharmaśāstras, and Agamas)
- Classical Indian texts (The Bhagavad Gita, The Śrīmad Bhāgavatam, and others)
- Indian historical texts (The Mahabharata and The Ramayana)
- Teachings of revered Saints, Sages, Siddhas, Yogis, Sadhus, Rishis, Gurus, Mystics, and Spiritual Masters of all genders, backgrounds, traditions, and walks of life

As stated in app.py:

> "Anveshak draws from a rich tapestry of spiritual wisdom found in classical Indian texts, philosophical treatises, and the teachings of revered Saints, Sages, Siddhas, Yogis, Sadhus, Rishis, Gurus, Mystics, and Spiritual Masters across centuries. The knowledge presented here spans multiple traditions, schools of thought, and spiritual lineages that have flourished in the Indian subcontinent and beyond."

### Ethical Sourcing

All texts included in Anveshak meet the following criteria:

1. **Public availability**: All texts were freely available from sources like archive.org
2. **Educational use**: Texts are used solely for educational purposes
3. **Proper attribution**: All sources are credited with author and publisher information
4. **Respect for copyright**: Implementation of word limits and other copyright-respecting measures

As mentioned in app.py:

> "Note that the sources consist of about 133 digitized texts, all of which were freely available over the internet (on sites like archive.org). Many of the texts are English translations of original (and in some cases, ancient) sacred and spiritual texts. All of the copyrights belong to the respective authors and publishers and we bow down in gratitude to their selfless work. Anveshak merely re-presents the ocean of spiritual knowledge and wisdom contained in the original works with relevant citations in a limited number of words."

## Data Processing Pipeline

### 1. Data Collection

The data collection process involves two methods as implemented in preprocessing.py:

#### Manual Upload
Texts are manually uploaded to Google Cloud Storage (GCS) through a preprocessing script:

```python
def upload_files_to_colab():
    """Upload raw text files and metadata from local machine to Colab."""
    # First, upload text files
    print("Step 1: Please upload your text files...")
    uploaded_text_files = files.upload()  # This will prompt the user to upload files

    # Create directory structure if it doesn't exist
    os.makedirs(LOCAL_RAW_TEXTS_FOLDER, exist_ok=True)

    # Move uploaded text files to the raw-texts folder
    for filename, content in uploaded_text_files.items():
        if filename.endswith(".txt"):
            with open(os.path.join(LOCAL_RAW_TEXTS_FOLDER, filename), "wb") as f:
                f.write(content)
            print(f"✅ Saved {filename} to {LOCAL_RAW_TEXTS_FOLDER}")
```

#### Web Downloading
Some texts are automatically downloaded from URLs listed in the metadata file:

```python
def download_text_files():
    """Fetch metadata, filter unuploaded files, and download text files."""
    metadata = fetch_metadata_from_gcs()
    # Filter entries where Uploaded is False
    files_to_download = [item for item in metadata if item["Uploaded"] == False]
    
    # Process only necessary files
    for item in files_to_download:
        name, author, url = item["Title"], item["Author"], item["URL"]
        if url.lower() == "not available":
            print(f"❌ Skipping {name} - No URL available.")
            continue

        try:
            response = requests.get(url)
            if response.status_code == 200:
                raw_text = response.text
                filename = "{}.txt".format(name.replace(" ", "_"))
                # Save to local first
                local_path = f"/tmp/{filename}"
                with open(local_path, "w", encoding="utf-8") as file:
                    file.write(raw_text)
                # Upload to GCS
                gcs_path = f"{RAW_TEXTS_DOWNLOADED_PATH_GCS}{filename}"
                upload_to_gcs(local_path, gcs_path)
                print(f"✅ Downloaded & uploaded: {filename} ({len(raw_text.split())} words)")
            else:
                print(f"❌ Failed to download {name}: {url} (Status {response.status_code})")
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
```

### 2. Text Cleaning

Raw texts often contain HTML tags, OCR errors, and formatting issues. The cleaning process removes these artifacts using the exact implementation from preprocessing.py:

```python
def rigorous_clean_text(text):
    """
    Clean text by removing metadata, junk text, and formatting issues.

    This function:
    1. Removes HTML tags using BeautifulSoup
    2. Removes URLs and standalone numbers
    3. Removes all-caps OCR noise words
    4. Deduplicates adjacent identical lines
    5. Normalizes Unicode characters
    6. Standardizes whitespace and newlines

    Args:
        text (str): The raw text to clean

    Returns:
        str: The cleaned text
    """
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"https?:\/\/\S+", "", text)  # Remove links
    text = re.sub(r"\b\d+\b", "", text)  # Remove standalone numbers
    text = re.sub(r"\b[A-Z]{5,}\b", "", text)  # Remove all-caps OCR noise words
    lines = text.split("\n")
    cleaned_lines = []
    last_line = None

    for line in lines:
        line = line.strip()
        if line and line != last_line:
            cleaned_lines.append(line)
            last_line = line

    text = "\n".join(cleaned_lines)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\n{2,}", "\n", text)
    return text
```

The cleaning process:
- Removes HTML tags using BeautifulSoup
- Eliminates URLs and standalone numbers
- Removes all-caps OCR noise words (common in digitized texts)
- Deduplicates adjacent identical lines
- Normalizes Unicode characters
- Standardizes whitespace and newlines

### 3. Text Chunking

Clean texts are split into smaller, manageable chunks for processing using the exact implementation from preprocessing.py:

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into smaller, overlapping chunks for better retrieval.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): Maximum number of words per chunk
        overlap (int): Number of words to overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks
```

Chunking characteristics:
- **Chunk size**: 500 words per chunk, balancing context and retrieval precision
- **Overlap**: 50-word overlap between chunks to maintain context across chunk boundaries
- **Context preservation**: Ensures that passages aren't arbitrarily cut in the middle of important concepts

### 4. Text Embedding

Chunks are converted to vector embeddings using the E5-large-v2 model with the actual implementation from preprocessing.py:

```python
def create_embeddings(text_chunks, batch_size=32):
    """
    Generate embeddings for the given chunks of text using the specified embedding model.
    
    This function:
    1. Uses SentenceTransformer to load the embedding model
    2. Prefixes each chunk with "passage:" as required by the E5 model
    3. Processes chunks in batches to manage memory usage
    4. Normalizes embeddings for cosine similarity search
    
    Args:
        text_chunks (list): List of text chunks to embed
        batch_size (int): Number of chunks to process at once
    
    Returns:
        numpy.ndarray: Matrix of embeddings, one per text chunk
    """
    # Load the model with GPU optimization
    model = SentenceTransformer(EMBEDDING_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"🚀 Using device for embeddings: {device}")

    prefixed_chunks = [f"passage: {text}" for text in text_chunks]
    all_embeddings = []

    for i in range(0, len(prefixed_chunks), batch_size):
        batch = prefixed_chunks[i:i+batch_size]
        # Move batch to GPU (if available) for faster processing
        with torch.no_grad():
            batch_embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(batch_embeddings)

        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(prefixed_chunks):
            print(f"📌 Processed {i + min(batch_size, len(prefixed_chunks) - i)}/{len(prefixed_chunks)} documents")

    return np.vstack(all_embeddings).astype("float32")
```

Embedding process details:
- **Model**: E5-large-v2, a state-of-the-art embedding model for retrieval tasks
- **Prefix**: "passage:" prefix is added to each chunk for optimal embedding
- **Batching**: Processing in batches of 32 for memory efficiency
- **Normalization**: Embeddings are normalized for cosine similarity search
- **Output**: Each text chunk becomes a 1024-dimensional vector

### 5. FAISS Index Creation

Embeddings are stored in a Facebook AI Similarity Search (FAISS) index for efficient similarity search:

```python
# Build FAISS index
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(all_embeddings)
```

FAISS index characteristics:
- **Index type**: IndexFlatIP (Inner Product) for cosine similarity search
- **Exact search**: Uses exact search rather than approximate for maximum accuracy
- **Dimension**: 1024-dimensional vectors from the E5-large-v2 model

### 6. Metadata Management

The system maintains metadata for each text to provide proper citations, using the implementation from rag_engine.py:

```python
def fetch_metadata_from_gcs():
    """
    Fetch metadata.jsonl from GCS and return as a list of dictionaries.
    
    Each dictionary represents a text entry with metadata like title, author, etc.
    
    Returns:
        list: List of dictionaries containing metadata for each text
    """
    blob = bucket.blob(METADATA_PATH_GCS)
    # Download metadata file
    metadata_jsonl = blob.download_as_text()
    # Parse JSONL
    metadata = [json.loads(line) for line in metadata_jsonl.splitlines()]
    return metadata
```

Metadata structure (JSONL format):
```json
{"Title": "Example Title 1", "Author": "Example Author 1", "Publisher": "Example Publisher 1", "URL": "https://example.com/1", "Uploaded": true}
{"Title": "Example Title 2", "Author": "Example Author 2", "Publisher": "Example Publisher 2", "URL": "https://example.com/2", "Uploaded": false}
```

## Data Storage Architecture

### Google Cloud Storage Structure

Anveshak: Spirituality Q&A uses Google Cloud Storage (GCS) as its primary data store, organized as follows:

```
bucket_name/
├── metadata/
│   └── metadata.jsonl           # Metadata for all texts
├── raw-texts/
│   ├── uploaded/                # Manually uploaded texts
│   └── downloaded/              # Automatically downloaded texts
├── cleaned-texts/               # Cleaned versions of all texts
└── processed/
    ├── embeddings/
    │   └── all_embeddings.npy   # Numpy array of embeddings
    ├── indices/
    │   └── faiss_index.faiss    # FAISS index file
    └── chunks/
        └── text_chunks.txt      # Text chunks with metadata
```

### Local Caching

For deployment on Hugging Face Spaces, essential files are downloaded to local storage using the implementation from rag_engine.py:

```python
# Local Paths
local_embeddings_file = "all_embeddings.npy"
local_faiss_index_file = "faiss_index.faiss"
local_text_chunks_file = "text_chunks.txt"
local_metadata_file = "metadata.jsonl"
```

These files are loaded with caching to improve performance, using the actual implementation from rag_engine.py:

```python
@st.cache_resource(show_spinner=False)
def cached_load_data_files():
    """
    Cached version of load_data_files() for FAISS index, text chunks, metadata, and OpenAI client.
    
    This function loads:
    - FAISS index for vector similarity search
    - Text chunks containing the original spiritual text passages
    - Metadata dictionary with publication and author information
    - OpenAI client instance for answer generation
    
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
    
    # Load FAISS index, text chunks, and metadata
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
```

## Data Access During Query Processing

### Query Embedding

User queries are embedded using the same model as the text corpus, with the actual implementation from rag_engine.py:

```python
def get_embedding(text):
    """
    Generate embeddings for a text query using the cached model.
    
    Uses an in-memory cache to avoid redundant embedding generation for repeated queries.
    Properly prefixes inputs with "query:" or "passage:" as required by the E5 model.
    
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
            return np.zeros((1, 384), dtype=np.float32)
            
        # Format input based on text length
        # For E5 models, "query:" prefix is for questions, "passage:" for documents
        input_text = f"query: {text}" if len(text) < 512 else f"passage: {text}"
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
        return np.zeros((1, 384), dtype=np.float32)
```

Note the use of:
- **Query prefix**: "query:" is added to distinguish query embeddings from passage embeddings
- **Truncation**: Queries are truncated to 512 tokens if necessary
- **Memory management**: Tensors are detached and moved to CPU after computation
- **Caching**: Query embeddings are cached to avoid redundant computation

### Passage Retrieval

The system retrieves relevant passages based on query embedding similarity using the implementation from rag_engine.py:

```python
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
```

Important aspects:
- **Similarity threshold**: Passages must have a similarity score >= 0.5 to be included
- **Diversity**: Only one passage per source title is included in the results
- **Metadata enrichment**: Publisher information is added from the metadata
- **Configurable retrieval**: The `top_k` parameter allows users to adjust how many sources to use

## User Data Privacy

### No Data Collection

Anveshak is designed to respect user privacy by not collecting or storing any user data:

1. **No Query Storage**: User questions are processed in memory and not saved
2. **No User Identification**: No user accounts or identification is required
3. **No Analytics**: No usage tracking or analytics are implemented
4. **No Cookies**: No browser cookies are used to track users

As stated in app.py:

> "We do not save any user data or queries. However, user questions are processed using OpenAI's LLM service to generate responses. While we do not store this information, please be aware that interactions are processed through OpenAI's platform and are subject to their privacy policies and data handling practices."

This privacy-first approach ensures that users can freely explore spiritual questions without concerns about their queries being stored or analyzed.

## Copyright and Ethical Considerations

### Word Limit Implementation

To respect copyright and ensure fair use, answers are limited to a configurable word count using the actual implementation from rag_engine.py:

```python
def answer_with_llm(query, openai_client, context=None, word_limit=200):
    # ... LLM processing ...
    
    # Extract and format the answer
    answer = response.choices[0].message.content.strip()
    words = answer.split()
    if len(words) > word_limit:
        answer = " ".join(words[:word_limit])
        if not answer.endswith((".", "!", "?")):
            answer += "."
            
    return answer
```

Users can adjust the word limit from 50 to 500 words, ensuring that responses are:
- Short enough to respect copyright
- Long enough to provide meaningful information
- Always properly cited to the original source

### Citation Format

Every answer includes citations to the original sources using the implementation from rag_engine.py:

```python
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
        if publisher.endswith(('.', '!', '?')):
            formatted_citations.append(f"📚 {title} by {author}, Published by {publisher}")
        else:
            formatted_citations.append(f"📚 {title} by {author}, Published by {publisher}.")
    return "\n".join(formatted_citations)
```

Citations include:
- Book/text title
- Author name
- Publisher information

### Acknowledgment of Sources

Anveshak: Spirituality Q&A includes dedicated pages for acknowledging:
- Publishers of the original texts
- Saints, Sages, and Spiritual Masters whose teachings are referenced
- The origins and traditions of the spiritual texts

A thank-you note is also prominently featured on the main page, as shown in app.py:

```python
st.markdown('<div class="acknowledgment-header">A Heartfelt Thank You</div>', unsafe_allow_html=True)
st.markdown("""
It is believed that one cannot be in a spiritual path without the will of the Lord. One need not be a believer or a non-believer, merely proceeding to thoughtlessness and observation is enough to evolve and shape perspectives. But that happens through grace. It is believed that without the will of the Lord, one cannot be blessed by real Saints, and without the will of the Saints, one cannot get close to them or God.

Therefore, with deepest reverence, we express our gratitude to:

**The Saints, Sages, Siddhas, Yogis, Sadhus, Rishis, Gurus, Mystics, and Spiritual Masters** of all genders, backgrounds, traditions, and walks of life whose timeless wisdom illuminates Anveshak. From ancient Sages to modern Masters, their selfless dedication to uplift humanity through selfless love and spiritual knowledge continues to guide seekers on the path.
# ...
""")
```

### Inclusive Recognition

Anveshak explicitly acknowledges and honors spiritual teachers from all backgrounds:

- All references to spiritual figures capitalize the first letter (Saints, Sages, etc.)
- The application includes language acknowledging Masters of "all genders, backgrounds, traditions, and walks of life"
- The selection of texts aims to represent diverse spiritual traditions

From the Sources.py file:

> "Additionally, there are and there have been many other great Saints, enlightened beings, Sadhus, Sages, and Gurus who have worked tirelessly to uplift humanity and guide beings to their true SELF and path, of whom little is known and documented. We thank them and acknowledge their contribution to the world."

## Data Replication and Backup

### GCS as Primary Storage

Google Cloud Storage serves as both the primary storage and backup system:

- All preprocessed data is stored in GCS buckets
- GCS provides built-in redundancy and backup capabilities
- Data is loaded from GCS at application startup

### Local Caching

For performance, Anveshak caches data locally using the implementation from rag_engine.py:

```python
def download_file_from_gcs(bucket, gcs_path, local_path):
    """
    Download a file from GCS to local storage if not already present.
    
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
```

This approach:
- Avoids redundant downloads
- Preserves data across application restarts
- Reduces API calls to GCS

## Conclusion

Anveshak: Spirituality Q&A implements a comprehensive data handling strategy that:

1. **Respects Copyright**: Through word limits, citations, and acknowledgments
2. **Preserves Source Integrity**: By maintaining accurate metadata and citations
3. **Optimizes Performance**: Through efficient storage, retrieval, and caching
4. **Ensures Ethical Use**: By focusing on educational purposes and proper attribution
5. **Protects Privacy**: By not collecting or storing user data
6. **Honors Diversity**: By acknowledging spiritual teachers of all backgrounds and traditions

This balance between technical efficiency and ethical responsibility allows Anveshak to serve as a bridge to spiritual knowledge while respecting the original sources, traditions, and user privacy. The system is designed not to replace personal spiritual inquiry but to supplement it by making traditional wisdom more accessible.

As stated in the conclusion of the blog post:

> "The core philosophy guiding this project is that while technology can facilitate access to spiritual knowledge, the journey to self-discovery remains deeply personal. As Anveshak states: 'The path and journey to the SELF is designed to be undertaken alone. The all-encompassing knowledge is internal and not external.'"