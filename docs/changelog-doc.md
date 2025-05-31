# Changelog

All notable changes to Anveshak: Spirituality Q&A will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-05-31
### Changed
- MUpdated default word limit from 100 to 200 words across all functions for consistency with UI slider
- MImproved OpenAI authentication priority order to prioritize HF Spaces environment variables
- MSimplified embedding generation logic to always use "query:" prefix for user queries
- MUpdated embedding model fallback dimensions from 384 to 1024 to match E5-large-v2

### Removed
- MRemoved unused CSS classes and styling rules across all pages
- MRemoved unnecessary text length condition for embedding prefix selection
- MCleaned up obsolete comments and documentation references

### Fixed
- MFixed inconsistency between function defaults and UI slider defaults for word limits
- MCorrected embedding model dimension mismatch in error handling

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