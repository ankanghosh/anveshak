# Configuration Guide

This document provides detailed instructions for configuring and deploying Anveshak: Spirituality Q&A, covering environment setup, authentication, customization options, and deployment strategies.

## Environment Configuration

### Configuration Parameters

Anveshak: Spirituality Q&A uses the following configuration parameters, which can be set through environment variables or Hugging Face Spaces secrets:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `BUCKET_NAME_GCS` | GCS bucket name for data storage | `"your-bucket-name"` |
| `METADATA_PATH_GCS` | Path to metadata file in GCS | `"metadata/metadata.jsonl"` |
| `EMBEDDINGS_PATH_GCS` | Path to embeddings file in GCS | `"processed/embeddings/all_embeddings.npy"` |
| `INDICES_PATH_GCS` | Path to FAISS index in GCS | `"processed/indices/faiss_index.faiss"` |
| `CHUNKS_PATH_GCS` | Path to text chunks file in GCS | `"processed/chunks/text_chunks.txt"` |
| `RAW_TEXTS_UPLOADED_PATH_GCS` | Path to uploaded raw texts in GCS | `"raw-texts/uploaded"` |
| `RAW_TEXTS_DOWNLOADED_PATH_GCS` | Path to downloaded raw texts in GCS | `"raw-texts/downloaded/"` |
| `CLEANED_TEXTS_PATH_GCS` | Path to cleaned texts in GCS | `"cleaned-texts/"` |
| `EMBEDDING_MODEL` | Hugging Face model ID for embeddings | `"intfloat/e5-large-v2"` |
| `LLM_MODEL` | OpenAI model for answer generation | `"gpt-3.5-turbo"` |
| `OPENAI_API_KEY` | OpenAI API key | `"sk-..."` |
| `GCP_CREDENTIALS` | GCP service account credentials (JSON) | `{"type":"service_account",...}` |

### Streamlit Secrets Configuration (Optional)

If developing locally with Streamlit, you can create a `.streamlit/secrets.toml` file with the following structure:

```toml
# GCS Configuration
BUCKET_NAME_GCS = "your-bucket-name"
METADATA_PATH_GCS = "metadata/metadata.jsonl"
EMBEDDINGS_PATH_GCS = "processed/embeddings/all_embeddings.npy"
INDICES_PATH_GCS = "processed/indices/faiss_index.faiss"
CHUNKS_PATH_GCS = "processed/chunks/text_chunks.txt"
RAW_TEXTS_UPLOADED_PATH_GCS = "raw-texts/uploaded"
RAW_TEXTS_DOWNLOADED_PATH_GCS = "raw-texts/downloaded/"
CLEANED_TEXTS_PATH_GCS = "cleaned-texts/"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
LLM_MODEL = "gpt-3.5-turbo"

# OpenAI API Configuration
openai_api_key = "your-openai-api-key"

# GCP Service Account Credentials (JSON format)
[gcp_credentials]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "your-private-key"
client_email = "your-client-email"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-client-cert-url"
```

### Environment Variables for Alternative Deployments

For deployments that support environment variables (like Heroku or Docker), you can use the following environment variables:

```bash
# GCS Configuration
export BUCKET_NAME_GCS="your-bucket-name"
export METADATA_PATH_GCS="metadata/metadata.jsonl"
export EMBEDDINGS_PATH_GCS="processed/embeddings/all_embeddings.npy"
export INDICES_PATH_GCS="processed/indices/faiss_index.faiss"
export CHUNKS_PATH_GCS="processed/chunks/text_chunks.txt"
export RAW_TEXTS_UPLOADED_PATH_GCS="raw-texts/uploaded"
export RAW_TEXTS_DOWNLOADED_PATH_GCS="raw-texts/downloaded/"
export CLEANED_TEXTS_PATH_GCS="cleaned-texts/"
export EMBEDDING_MODEL="intfloat/e5-large-v2"
export LLM_MODEL="gpt-3.5-turbo"

# OpenAI API Configuration
export OPENAI_API_KEY="your-openai-api-key"

# GCP Service Account (as a JSON string)
export GCP_CREDENTIALS='{"type":"service_account","project_id":"your-project-id",...}'
```

## Authentication Setup

### Google Cloud Storage (GCS) Authentication

Anveshak: Spirituality Q&A supports multiple methods for authenticating with GCS:

#### Setting Up a GCP Service Account (Required)

Before configuring authentication methods, you'll need to create a Google Cloud Platform (GCP) service account:

1. **Create a GCP project** (if you don't already have one):
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Click on "Select a project" at the top right and then "New Project"
   - Enter a project name and click "Create"

2. **Enable the Cloud Storage API**:
   - Go to "APIs & Services" > "Library" in the left sidebar
   - Search for "Cloud Storage"
   - Click on "Cloud Storage API" and then "Enable"

3. **Create a service account**:
   - Go to "IAM & Admin" > "Service Accounts" in the left sidebar
   - Click "Create Service Account"
   - Enter a service account name and description
   - Click "Create and Continue"

4. **Assign roles to the service account**:
   - Add the "Storage Object Admin" role for access to GCS objects
   - Add the "Viewer" role for basic read permissions
   - Click "Continue" and then "Done"

5. **Create and download service account key**:
   - Find your new service account in the list and click on it
   - Go to the "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON" as the key type
   - Click "Create" to download the key file (This is your GCP credentials JSON file)

6. **Create a GCS bucket**:
   - Go to "Cloud Storage" > "Buckets" in the left sidebar
   - Click "Create"
   - Enter a globally unique bucket name
   - Choose your settings for location, class, and access control
   - Click "Create"

Once you have created your service account and GCS bucket, you can use any of the following authentication methods:

#### Option 1: HF Spaces Environment Variable (Recommended Production Method)

For Hugging Face Spaces, set the `GCP_CREDENTIALS` environment variable in the Spaces UI:

1. Go to your Space settings
2. Under "Repository secrets"
3. Add a new secret with name `GCP_CREDENTIALS` and value containing your JSON credentials

#### Option 2: Local Development with Application Default Credentials

For local development, you can use Application Default Credentials:

```bash
# Export path to your service account key file
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-file.json"
```

#### Option 3: Streamlit Secrets

Add your service account credentials to the `.streamlit/secrets.toml` file as shown in the example above.

The authentication logic is handled by the `setup_gcp_auth()` function in `utils.py`:

```python
def setup_gcp_auth():
    """
    Setup Google Cloud Platform (GCP) authentication using various methods.
    
    This function tries multiple authentication methods in order of preference:
    1. HF Spaces environment variable (GCP_CREDENTIALS) - primary production method
    2. Local environment variable pointing to credentials file (GOOGLE_APPLICATION_CREDENTIALS)
    3. Streamlit secrets (gcp_credentials)
    
    Note: In production, credentials are stored exclusively in HF Spaces secrets.
    """
    try:
        # Option 1: HF Spaces environment variable
        if "GCP_CREDENTIALS" in os.environ:
            gcp_credentials = json.loads(os.getenv("GCP_CREDENTIALS"))
            print("✅ Using GCP credentials from HF Spaces environment variable")
            credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
            return credentials
            
        # Option 2: Local environment variable pointing to file
        elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            credentials_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            print(f"✅ Using GCP credentials from file at {credentials_path}")
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            return credentials
            
        # Option 3: Streamlit secrets
        elif "gcp_credentials" in st.secrets:
            gcp_credentials = st.secrets["gcp_credentials"]
            
            # Handle different secret formats
            if isinstance(gcp_credentials, dict) or hasattr(gcp_credentials, 'to_dict'):
                # Convert AttrDict to dict if needed
                if hasattr(gcp_credentials, 'to_dict'):
                    gcp_credentials = gcp_credentials.to_dict()
                    
                print("✅ Using GCP credentials from Streamlit secrets (dict format)")
                credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
                return credentials
            else:
                # Assume it's a JSON string
                try:
                    gcp_credentials_dict = json.loads(gcp_credentials)
                    print("✅ Using GCP credentials from Streamlit secrets (JSON string)")
                    credentials = service_account.Credentials.from_service_account_info(gcp_credentials_dict)
                    return credentials
                except json.JSONDecodeError:
                    print("⚠️ GCP credentials in Streamlit secrets is not valid JSON, trying as file path")
                    if os.path.exists(gcp_credentials):
                        credentials = service_account.Credentials.from_service_account_file(gcp_credentials)
                        return credentials
                    else:
                        raise ValueError("GCP credentials format not recognized")
                        
        else:
            raise ValueError("No GCP credentials found in environment or Streamlit secrets")
            
    except Exception as e:
        error_msg = f"❌ Authentication error: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        raise
```

### OpenAI API Authentication

Similarly, OpenAI API authentication can be configured in multiple ways:

#### Option 1: HF Spaces Environment Variable (Recommended Production Method)

Set the `OPENAI_API_KEY` environment variable in the Hugging Face Spaces UI.

#### Option 2: Environment Variables

Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### Option 3: Streamlit Secrets

Add your OpenAI API key to the `.streamlit/secrets.toml` file:

```toml
openai_api_key = "your-openai-api-key"
```

The authentication logic is handled by the `setup_openai_auth()` function in `utils.py`:

```python
def setup_openai_auth():
    """
    Setup OpenAI API authentication using various methods.
    
    This function tries multiple authentication methods in order of preference:
    1. Standard environment variable (OPENAI_API_KEY)
    2. HF Spaces environment variable (OPENAI_KEY) - primary production method
    3. Streamlit secrets (openai_api_key)
    
    Note: In production, the API key is stored exclusively in HF Spaces secrets.
    """
    try:
        # Option 1: Standard environment variable
        if "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            print("✅ Using OpenAI API key from environment variable")
            return
            
        # Option 2: HF Spaces environment variable with different name
        elif "OPENAI_KEY" in os.environ:
            openai.api_key = os.getenv("OPENAI_KEY")
            print("✅ Using OpenAI API key from HF Spaces environment variable")
            return
            
        # Option 3: Streamlit secrets
        elif "openai_api_key" in st.secrets:
            openai.api_key = st.secrets["openai_api_key"]
            print("✅ Using OpenAI API key from Streamlit secrets")
            return
            
        else:
            raise ValueError("No OpenAI API key found in environment or Streamlit secrets")
            
    except Exception as e:
        error_msg = f"❌ OpenAI authentication error: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        raise
```

## Application Customization

### UI Customization

Anveshak's UI can be customized through the CSS in the `app.py` file:

```python
# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    color: #c0392b;
    text-align: center;
    margin-bottom: 1rem;
}
.subtitle {
    font-size: 1.2rem;
    color: #555;
    text-align: center;
    margin-bottom: 1.5rem;
    font-style: italic;
}
/* More CSS rules... */
</style>
<div class="main-title">Anveshak</div>
<div class="subtitle">Spirituality Q&A</div>
""", unsafe_allow_html=True)
```

To change the appearance:

1. Modify the CSS variables in the `<style>` tag
2. Update color schemes, fonts, or layouts as needed
3. Add new CSS classes for additional UI elements

### Common Questions Configuration

The list of pre-selected common questions can be modified in the `app.py` file:

```python
# Common spiritual questions for users to select from
common_questions = [
    "What is the Atman or the soul?",
    "Are there rebirths?",
    "What is Karma?",
    # Add or modify questions here
]
```

### Retrieval Parameters

Two key retrieval parameters can be adjusted by users through the UI:

1. **Number of sources** (`top_k`): Controls how many distinct sources are used for generating answers
   - Default: 5
   - Range: 3-10
   - UI Component: Slider in the main interface

2. **Word limit** (`word_limit`): Controls the maximum length of generated answers
   - Default: 200
   - Range: 50-500
   - UI Component: Slider in the main interface

These parameters are implemented in the Streamlit UI:

```python
# Sliders for customization
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Number of sources:", 3, 10, 5)
with col2:
    word_limit = st.slider("Word limit:", 50, 500, 200)
```

## Deployment Options

### Recommended: Hugging Face Spaces Deployment

The recommended and tested deployment method for Anveshak: Spirituality Q&A is Hugging Face Spaces, which provides the necessary resources for running the application efficiently.

To deploy on Hugging Face Spaces:

1. Fork the repository to your GitHub account

2. Create a new Space on Hugging Face:
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select "Streamlit" as the SDK
   - Connect your GitHub repository

3. Configure secrets in the Hugging Face UI:
   - Go to your Space settings
   - Under "Repository secrets"
   - Add each of the following secrets:
     - `OPENAI_API_KEY`
     - `GCP_CREDENTIALS` (the entire JSON as a string)
     - `BUCKET_NAME_GCS`
     - `LLM_MODEL`
     - `METADATA_PATH_GCS`
     - `RAW_TEXTS_UPLOADED_PATH_GCS`
     - `RAW_TEXTS_DOWNLOADED_PATH_GCS`
     - `CLEANED_TEXTS_PATH_GCS`
     - `EMBEDDINGS_PATH_GCS`
     - `INDICES_PATH_GCS`
     - `CHUNKS_PATH_GCS`
     - `EMBEDDING_MODEL`

4. The app should automatically deploy. If needed, manually trigger a rebuild from the Spaces UI.

### Local Development (Not Recommended)

**Important Note**: Running Anveshak: Spirituality Q&A locally requires above 16GB of RAM due to the embedding model. Most standard laptops will experience crashes during model loading. Hugging Face Spaces deployment is strongly recommended.

If you still want to run it locally for development purposes:

1. Clone the repository
   ```bash
   git clone https://github.com/YourUsername/anveshak.git
   cd anveshak
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Create the `.streamlit/secrets.toml` file as described above

4. Run the application
   ```bash
   streamlit run app.py
   ```

### Alternative: Docker Deployment

For containerized deployment (not tested in production):

1. Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

2. Build the Docker image:
```bash
docker build -t anveshak .
```

3. Run the container:
```bash
docker run -p 8501:8501 \
  -e BUCKET_NAME_GCS=your-bucket-name \
  -e METADATA_PATH_GCS=metadata/metadata.jsonl \
  -e EMBEDDINGS_PATH_GCS=processed/embeddings/all_embeddings.npy \
  -e INDICES_PATH_GCS=processed/indices/faiss_index.faiss \
  -e CHUNKS_PATH_GCS=processed/chunks/text_chunks.txt \
  -e RAW_TEXTS_UPLOADED_PATH_GCS=raw-texts/uploaded \
  -e RAW_TEXTS_DOWNLOADED_PATH_GCS=raw-texts/downloaded/ \
  -e CLEANED_TEXTS_PATH_GCS=cleaned-texts/ \
  -e EMBEDDING_MODEL=intfloat/e5-large-v2 \
  -e LLM_MODEL=gpt-3.5-turbo \
  -e OPENAI_API_KEY=your-openai-api-key \
  -e GCP_CREDENTIALS='{"type":"service_account",...}' \
  anveshak
```

## Performance Tuning

### Caching Configuration

Anveshak: Spirituality Q&A uses Streamlit's caching mechanisms to optimize performance:

#### Resource Caching
Used for loading models and data files that remain constant:

```python
@st.cache_resource(show_spinner=False)
def cached_load_model():
    # Load embedding model once and cache it
```

This cache persists for the lifetime of the application.

#### Data Caching
Used for caching query results with a time-to-live (TTL):

```python
@st.cache_data(ttl=3600, show_spinner=False)
def cached_process_query(query, top_k=5, word_limit=100):
    # Cache query results for an hour
```

The TTL (3600 seconds = 1 hour) can be adjusted based on your needs.

### Memory Optimization

For deployments with limited memory:

1. **Force CPU Usage**: Anveshak already forces CPU usage for the embedding model to avoid GPU memory issues:
   ```python
   os.environ["CUDA_VISIBLE_DEVICES"] = ""
   ```

2. **Adjust Batch Size**: If you're recreating the embeddings, consider reducing the batch size:
   ```python
   def create_embeddings(text_chunks, batch_size=16):  # Reduced from 32
   ```

3. **Garbage Collection**: Anveshak performs explicit garbage collection after operations:
   ```python
   del outputs, inputs
   gc.collect()
   ```

## Troubleshooting

### Common Issues

#### Authentication Errors

**Symptom**: Error message about invalid credentials or permission denied.

**Solution**:
1. Verify that your service account has the correct permissions (Storage Object Admin)
2. Check that your API keys are correctly formatted and not expired
3. Ensure that your GCP credentials JSON is valid and properly formatted

#### Missing Files

**Symptom**: Error about missing files or "File not found" when accessing GCS.

**Solution**:
1. Verify the correct bucket name and file paths in your configuration
2. Check that all required files exist in your GCS bucket
3. Ensure your service account has access to the specified bucket

#### Memory Issues

**Symptom**: Application crashes with out-of-memory errors.

**Solution**:
1. Increase the memory allocation for your deployment (if possible)
2. Ensure that `os.environ["CUDA_VISIBLE_DEVICES"] = ""` is set to force CPU usage
3. Implement additional garbage collection calls in high-memory operations

#### OpenAI API Rate Limits

**Symptom**: Errors about rate limits or exceeding quotas with OpenAI.

**Solution**:
1. Implement retry logic with exponential backoff
2. Consider using a paid tier OpenAI account with higher rate limits
3. Add caching to reduce the number of API calls

### Logs and Debugging

Anveshak includes comprehensive logging:

```python
print(f"✅ Model loaded successfully (cached)")
print(f"❌ Error loading model: {str(e)}")
```

To enable more detailed logging, you can use Streamlit's built-in logging configuration:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Then use logger instead of print
logger.info("Model loaded successfully")
logger.error(f"Error loading model: {str(e)}")
```

## Special Considerations

### Privacy

Anveshak: Spirituality Q&A is designed to not save or store any user queries or data. This is important for spiritual questions, which may be of a personal nature. No additional configuration is needed for this - the application simply does not implement any data storage functionality.

### Language Support

Currently, Anveshak is only available in English. This is a known limitation of the current implementation. Future versions may include support for Sanskrit, Hindi, Bengali, Tamil, and other Indian languages.

### Concise Answers

Anveshak generates concise answers rather than lengthy explanations. This is by design, to respect both copyright constraints and the nature of spiritual wisdom, which often benefits from clarity and simplicity.

## Conclusion

This configuration guide provides all the necessary information to set up, customize, and deploy Anveshak: Spirituality Q&A. By following these instructions, you should be able to:

1. Configure the necessary authentication for GCS and OpenAI
2. Customize Anveshak's appearance and behavior
3. Deploy the application on Hugging Face Spaces (recommended) or other platforms
4. Optimize performance for your specific use case
5. Troubleshoot common issues

The flexibility of the configuration options allows you to adapt the application to different deployment environments while maintaining the core functionality of providing spiritually informed answers based on traditional texts from diverse traditions and teachers of all backgrounds.