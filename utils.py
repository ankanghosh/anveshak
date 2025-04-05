import os
import json
from google.oauth2 import service_account
import streamlit as st
import openai

def setup_gcp_auth():
    """
    Setup Google Cloud Platform (GCP) authentication using various methods.
    
    This function tries multiple authentication methods in order of preference:
    1. HF Spaces environment variable (GCP_CREDENTIALS) - this is our primary method, 
       with credentials stored exclusively in HF Spaces secrets for deployment
    2. Local environment variable pointing to a credentials file (GOOGLE_APPLICATION_CREDENTIALS) - 
       supported for development environments only
    3. Streamlit secrets (gcp_credentials) - fallback method
    
    Note: In production, we exclusively use HF Spaces secrets for storing credentials.
    The other methods are included for development flexibility and completeness.
    
    Returns:
        google.oauth2.service_account.Credentials: GCP credentials object
        
    Raises:
        ValueError: If no valid credentials are found
        Exception: For any authentication errors
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

def setup_openai_auth():
    """
    Setup OpenAI API authentication using various methods.
    
    This function tries multiple authentication methods in order of preference:
    1. Standard environment variable (OPENAI_API_KEY)
    2. HF Spaces environment variable (OPENAI_KEY) - this is our primary method,
       with API key stored exclusively in HF Spaces secrets for deployment
    3. Streamlit secrets (openai_api_key) - fallback method
    
    Note: In production, we exclusively use HF Spaces secrets for storing the OpenAI API key.
    The other methods are included for development flexibility and completeness.
    
    Raises:
        ValueError: If no valid API key is found
        Exception: For any authentication errors
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

def setup_all_auth():
    """
    Setup all authentication in one call.
    
    This is a convenience function that sets up both GCP and OpenAI authentication.
    It's typically called during Anveshak: Spirituality Q&A's initialization.
    
    In production, authentication credentials are stored exclusively in HF Spaces secrets.
    
    Returns:
        google.oauth2.service_account.Credentials: GCP credentials object
    """
    gcp_creds = setup_gcp_auth()
    setup_openai_auth()
    return gcp_creds