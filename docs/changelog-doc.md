# Changelog

All notable changes to Anveshak: Spirituality Q&A will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2025-10-25

### Fixed
- Fixed OpenAI SDK compatibility issues with HF Spaces proxy environment by adding explicit httpx client handling
- Resolved torch float16 CPU incompatibility by switching to bfloat16 for better CPU support
- Fixed dependency version conflicts that caused application crashes by pinning all package versions
- Fixed button text truncation in UI by updating CSS to allow multi-line text display

### Changed
- Pinned all package versions in requirements.txt for production stability (torch==2.2.1, transformers==4.38.2, sentence-transformers==2.5.1, openai==1.51.0)
- Updated OpenAI client initialization to use httpx for proper proxy handling in HF Spaces environment
- Changed model loading to use bfloat16 instead of float16 for improved CPU compatibility
- Updated button CSS styling to wrap text instead of truncating with ellipsis
- Modified common question text: "What was Swami Vivekananda's opinion about the SELF?" to "Swami Vivekananda's opinion on SELF?" for better UI fit

### Added
- Added httpx>=0.24.0 to requirements.txt for explicit proxy handling
- Added OpenAI client as a parameter passed through the entire RAG pipeline
- Added numpy>=1.26.0 version constraint for Python 3.12 compatibility during local development

### Technical Details
- OpenAI client now initialized once with httpx.Client() and cached via @st.cache_resource
- OpenAI client instance passed as parameter to answer_with_llm() function
- cached_load_data_files() now returns tuple of (faiss_index, text_chunks, metadata_dict, openai_client)
- Model loading uses torch.bfloat16 instead of torch.float16 for CPU operations
- Updated architecture documentation to reflect OpenAI client pattern throughout codebase

## [1.0.1] - 2025-05-31

### Changed
- Updated default word limit from 100 to 200 words across all functions for consistency with UI slider
- Improved OpenAI authentication priority order to prioritize HF Spaces environment variables
- Simplified embedding generation logic to always use "query:" prefix for user queries
- Updated embedding model fallback dimensions from 384 to 1024 to match E5-large-v2

### Removed
- Removed unused CSS classes and styling rules across all pages
- Removed unnecessary text length condition for embedding prefix selection
- Cleaned up obsolete comments and documentation references

### Fixed
- Fixed inconsistency between function defaults and UI slider defaults for word limits
- Corrected embedding model dimension mismatch in error handling

## [1.0.0] - 2025-04-01

### Added
- Initial release of Anveshak: Spirituality Q&A
- Core RAG functionality with E5-large-v2 embedding model
- FAISS index for efficient text retrieval
- Integration with OpenAI API for answer generation
- Streamlit-based user interface
- Caching mechanisms for improved performance
- Support for customizable number of sources and word limits
- Pre-selected common spiritual questions
- Comprehensive acknowledgment of sources and publishers
- Detailed documentation

### Technical Features
- Google Cloud Storage integration for data storage
- Authentication handling for GCP and OpenAI
- Memory optimization for resource-constrained environments
- Multi-page Streamlit application structure
- Custom CSS styling for enhanced user experience
- Privacy protection with no user data storage
- Concise answer generation system
- Recognition of Saints and Spiritual Masters of all backgrounds and traditions

## Future Roadmap

### Planned for v1.1.0
- Multi-language support (Sanskrit, Hindi, Bengali, Tamil, and more)
- User feedback collection for answer quality
- Enhanced answer relevance with hybrid retrieval methods
- Additional spiritual texts from diverse traditions
- Improved citation formatting with page numbers where available

### Planned for v1.2.0
- Self-hosted open-source LLM integration
- Advanced visualization of concept relationships
- Search functionality for specific texts or authors
- Audio output for visually impaired users
- Mobile-optimized interface

### Planned for v2.0.0
- Meditation timer and guide integration
- Personalized learning paths based on user interests (implemented with privacy-preserving approaches like client-side storage, session-based preferences, or explicit opt-in)
- Interactive glossary of spiritual terms
- Spiritual practice guide with scheduler and tracker
- Community features for discussion and shared learning