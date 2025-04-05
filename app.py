import streamlit as st
import time
import streamlit.components.v1 as components

# FIRST: Set page config before ANY other Streamlit command
st.set_page_config(page_title="Anveshak: Spirituality Q&A", page_icon="🕉️")

# Initialize ALL session state variables right at the beginning
# This ensures consistent state management across application reruns
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'submit_clicked' not in st.session_state:
    st.session_state.submit_clicked = False
if 'init_time' not in st.session_state:
    st.session_state.init_time = None
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0  # This will help us reset the form
# New variable for debouncing: whether processing is in progress
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
# Store the answer so that it persists on screen
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = None
# Add new session state for showing/hiding acknowledgment
if 'show_acknowledgment' not in st.session_state:
    st.session_state.show_acknowledgment = False
# Add page change detection
if 'page_loaded' not in st.session_state:
    st.session_state.page_loaded = True
    # Reset query state when returning to home page
    st.session_state.last_query = ""
    st.session_state.last_answer = None

# THEN: Import your modules
from rag_engine import process_query, load_model, cached_load_data_files
from utils import setup_all_auth

# Function to toggle acknowledgment visibility
def toggle_acknowledgment():
    """Toggle the visibility of the acknowledgment section."""
    st.session_state.show_acknowledgment = not st.session_state.show_acknowledgment

# Custom HTML/JS to navigate to sources page
def navigate_to_sources():
    """
    Create a JavaScript function to navigate to the Sources page.
    
    This uses Streamlit components to inject JavaScript that finds and clicks
    the Sources link in the sidebar navigation.
    """
    components.html(
        """
        <script>
        // Wait for the page to fully load
        document.addEventListener('DOMContentLoaded', (event) => {
            // This select the nav item for the Sources page in the sidebar
            const sourcesLink = Array.from(document.querySelectorAll('a.css-z5fcl4')).find(el => el.innerText === 'Sources');
            if (sourcesLink) {
                sourcesLink.click();
            }
        });
        </script>
        """,
        height=0,
    )

# Custom styling (pure CSS)
st.markdown("""
<style>
/* Button styling */
.stButton>button {
    background-color: #fff0f0 !important;
    color: #3f51b5 !important;
    border: 1px solid #e1e4f2 !important;
    border-radius: 20px !important;
    padding: 8px 16px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
/* Form submit button specific styling */
button[type="submit"], 
.stFormSubmit>button, 
[data-testid="stFormSubmitButton"]>button {
    background-color: #fff0f0 !important;
    color: #3f51b5 !important;
    border: 1px solid #e1e4f2 !important;
    border-radius: 8px !important;
}
.stButton>button:hover {
    background-color: #fafbff !important;
    border-color: #c5cae9 !important;
}
/* Special styling for thank you button */
.thank-you-button > button {
    background-color: #f8e6ff !important;
    border-radius: 8px !important; 
    padding: 10px 20px !important;
    font-weight: normal !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    transition: all 0.3s ease !important;
    border: 1px solid #d8b9ff !important;
    color: #6a1b9a !important;
}
.thank-you-button > button:hover {
    background-color: #f0d6ff !important;
    transform: translateY(-1px) !important;
}
/* Input field styling */
div[data-baseweb="input"] {
    border: 1px solid #fff0f0 !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
}
div[data-baseweb="input"]:focus-within {
    border: 1px solid #3f51b5 !important;
}
div[data-baseweb="input"]:active {
    border: 1px solid #fff0f0 !important;
}
/* Style the st.info boxes */
div.stInfo {
    background-color: #f8faff !important;
    color: #3f51b5 !important;
    border: 1px solid #e1e4f2 !important;
    border-radius: 8px !important;
}
/* COMBINED SCROLL CONTAINER */
.questions-scroll-container {
    width: 100%;
    overflow-x: auto;
    scrollbar-width: none; /* Firefox */
    -ms-overflow-style: none; /* IE and Edge */
}
/* Hide scrollbar for Chrome, Safari and Opera */
.questions-scroll-container::-webkit-scrollbar {
    display: none;
}
/* Inner content that holds both rows */
.questions-content {
    display: inline-flex;
    flex-direction: column;
    min-width: max-content;
    gap: 10px;
    padding: 5px 0;
}
/* Individual rows */
.questions-row {
    display: flex;
    flex-direction: row;
    gap: 10px;
}
/* Placeholder for buttons */
.button-placeholder {
    min-height: 38px;
    min-width: 120px;
    margin: 0 5px;
}
/* Acknowledgment section styling - fully fixed */
.acknowledgment-container {
    background-color: #f8f5ff;
    border: 1px solid #e0d6fe;
    border-radius: 8px;
    padding: 15px;
    margin: 20px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.acknowledgment-header {
    color: #6a1b9a;
    font-size: 1.3rem;
    margin-bottom: 10px;
    text-align: center;
}
.more-info-link {
    text-align: center;
    margin-top: 10px;
    font-style: italic;
}
.citation-note {
    font-size: 0.8rem;
    font-style: italic;
    color: #666;
    padding: 10px;
    border-top: 1px solid #eee;
    margin-top: 30px;
}
/* Center align title */
.main-title {
    font-size: 2.5rem;
    color: #c0392b;
    text-align: center;
    margin-bottom: 1rem;
}
/* Subtitle styling */
.subtitle {
    font-size: 1.2rem;
    color: #555;
    text-align: center;
    margin-bottom: 1.5rem;
    font-style: italic;
}
/* Button container for centering */
.center-container {
    display: flex;
    justify-content: center;
    margin: 0 auto;
    width: 100%;
}
/* Source link styling */
.source-link {
    color: #3f51b5;
    font-weight: bold;
    text-decoration: underline;
    cursor: pointer;
}
.source-link:hover {
    color: #6a1b9a;
}
</style>
<div class="main-title">Anveshak</div>
<div class="subtitle">Spirituality Q&A</div>
""", unsafe_allow_html=True)

# Centered button layout without columns
_, center_col, _ = st.columns([1, 2, 1])  # Create a wider center column
with center_col:  # Put everything in the center column
    if st.session_state.show_acknowledgment:
        st.markdown('<div class="thank-you-button">', unsafe_allow_html=True)
        if st.button("Hide Thank You Note", on_click=toggle_acknowledgment, disabled=st.session_state.is_processing, use_container_width=True):
            pass
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="thank-you-button">', unsafe_allow_html=True)
        if st.button("Show Thank You Note", on_click=toggle_acknowledgment, disabled=st.session_state.is_processing, use_container_width=True):
            pass
        st.markdown('</div>', unsafe_allow_html=True)

# Preload resources during initialization
# This ensures the model, index, and data are loaded before the user interacts with the app
init_message = st.empty()
if not st.session_state.initialized:
    init_message.info("Hang in there! We are setting the system up for you. 😊")
    try:
        # Setup authentication and preload heavy resources
        setup_all_auth()
        load_model()  # This uses cached_load_model via alias
        cached_load_data_files()  # Preload FAISS index, text chunks, and metadata
        st.session_state.initialized = True
        st.session_state.init_time = time.time()
        init_message.success("System initialized successfully!")
        time.sleep(2)
        init_message.empty()
    except Exception as e:
        init_message.error(f"Error initializing: {str(e)}")
elif st.session_state.init_time is not None:
    elapsed_time = time.time() - st.session_state.init_time
    if elapsed_time >= 2.0:
        init_message.empty()
        st.session_state.init_time = None

# Display the acknowledgment section if toggled on
if st.session_state.show_acknowledgment:
    st.markdown('<div class="acknowledgment-container">', unsafe_allow_html=True)
    st.markdown('<div class="acknowledgment-header">A Heartfelt Thank You</div>', unsafe_allow_html=True)
    st.markdown("""
    It is believed that one cannot be in a spiritual path without the will of the Lord. One need not be a believer or a non-believer, merely proceeding to thoughtlessness and observation is enough to evolve and shape perspectives. But that happens through grace. It is believed that without the will of the Lord, one cannot be blessed by real Saints, and without the will of the Saints, one cannot get close to them or God.
    
    Therefore, with deepest reverence, we express our gratitude to:
    
    **The Saints, Sages, Siddhas, Yogis, Sadhus, Rishis, Gurus, Mystics, and Spiritual Masters** of all genders, backgrounds, traditions, and walks of life whose timeless wisdom illuminates Anveshak. From ancient Sages to modern Masters, their selfless dedication to uplift humanity through selfless love and spiritual knowledge continues to guide seekers on the path.
    
    **The Sacred Texts** that have preserved the eternal truths across millennia, offering light in times of darkness and clarity in times of confusion.
    
    **The Publishers** who have diligently preserved and disseminated these precious teachings, making them accessible to spiritual aspirants worldwide. Their work ensures these wisdom traditions endure for future generations.
    
    **The Authors** who have dedicated their lives to interpreting and explaining complex spiritual concepts, making them accessible to modern readers.
    
    This application is merely a humble vessel for the ocean of wisdom they have shared with the world. We claim no ownership of these teachings - only profound gratitude for the opportunity to help make them more accessible.
    """)
    
    # Link to Sources using Streamlit's built-in way
    st.markdown('<div class="more-info-link">', unsafe_allow_html=True)
    st.write("For detailed information about our sources, please visit the *Sources* page in the navigation menu.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to handle query selection from the common questions buttons
def set_query(query):
    """
    Handle selection of a pre-defined question from the UI.
    
    This sets the selected question as the current query, marks it for processing,
    and triggers a rerun of the app to process the query.
    
    Args:
        query (str): The selected question text
    """
    # If already processing, ignore further input to prevent multiple submissions
    if st.session_state.is_processing:
        return
    st.session_state.last_query = query
    st.session_state.submit_clicked = True
    st.session_state.is_processing = True
    st.rerun()

# Function to group questions into rows based on length
def group_buttons(questions, max_chars_per_row=100):
    """
    Organize common questions into rows for better UI layout.
    
    This function groups questions into rows based on character length to
    ensure they fit well in the UI and look visually balanced.
    
    Args:
        questions (list): List of question strings
        max_chars_per_row (int): Maximum total characters per row
        
    Returns:
        list: A list of rows, where each row is a list of questions
    """
    rows = []
    current_row = []
    current_length = 0
    for q in questions:
        # Add some buffer for button padding/margins
        q_length = len(q) + 5
        if current_length + q_length > max_chars_per_row and current_row:
            rows.append(current_row)
            current_row = [q]
            current_length = q_length
        else:
            current_row.append(q)
            current_length += q_length
    if current_row:
        rows.append(current_row)
    return rows

# Common spiritual questions for users to select from
# These represent fundamental spiritual inquiries across traditions
common_questions = [
    "What is the Atman or the soul?",
    "Are there rebirths?",
    "What is Karma?",
    "What is the ultimate truth?",
    "What is Swami Vivekananda's opinion about the SELF?",
    "Explain moksha or salvation. Is that for real?",
    "Destiny or free will?",
    "What is the ultimate goal of human life?",
    "Do we really die?",
    "How can you attain self-realization?"
]

# Display heading for common questions
st.markdown("### Few questions to try:")

# Group questions into rows and create buttons (disabled if processing)
question_rows = group_buttons(common_questions, max_chars_per_row=80)
for row_idx, row in enumerate(question_rows):
    cols = st.columns(len(row))
    for i, (col, q) in enumerate(zip(cols, row)):
        with col:
            if st.button(q, key=f"r{row_idx}_q{i}", use_container_width=True, disabled=st.session_state.is_processing):
                set_query(q)

# Function to handle custom question submission from the form
def handle_form_submit():
    """
    Process user-submitted questions from the input form.
    
    This function:
    1. Checks if a query is already being processed
    2. Validates that the input isn't empty
    3. Sets the query for processing
    4. Increments the form key to force a reset after submission
    """
    # If already processing, ignore further input to prevent multiple submissions
    if st.session_state.is_processing:
        return
    if st.session_state.query_input and st.session_state.query_input.strip():
        st.session_state.last_query = st.session_state.query_input.strip()
        st.session_state.submit_clicked = True
        st.session_state.is_processing = True
        # Increment the form key to force a reset
        st.session_state.form_key += 1

# Create a form with a dynamic key (to allow resetting after submission)
with st.form(key=f"query_form_{st.session_state.form_key}"):
    query = st.text_input("Ask your question:", key="query_input", 
                          placeholder="Press enter to submit your question", disabled=st.session_state.is_processing)
    submit_button = st.form_submit_button("Get Answer", on_click=handle_form_submit, disabled=st.session_state.is_processing)

# Display the current question if available
if st.session_state.last_query:
    st.markdown("### Current Question:")
    st.info(st.session_state.last_query)

# Sliders for customization - allows users to control retrieval parameters
col1, col2 = st.columns(2)
with col1:
    # Control how many different sources will be used for the answer
    top_k = st.slider("Number of sources:", 3, 10, 5)
with col2:
    # Control the maximum length of the generated answer
    word_limit = st.slider("Word limit:", 50, 500, 200)

# Process the query only if it has been explicitly submitted
# This prevents automatic processing on page load or slider adjustments
if st.session_state.submit_clicked and st.session_state.last_query:
    st.session_state.submit_clicked = False
    with st.spinner("Processing your question..."):
        try:
            # Call the RAG engine with the user's question and retrieval parameters
            result = process_query(st.session_state.last_query, top_k=top_k, word_limit=word_limit)
            st.session_state.last_answer = result  # Store result in session state
        except Exception as e:
            st.session_state.last_answer = {"answer_with_rag": f"Error processing query: {str(e)}", "citations": ""}
    # Reset debouncing after processing and force a rerun to re-enable buttons
    st.session_state.is_processing = False
    st.rerun()

# Display the answer and citations if available
if st.session_state.last_answer is not None:
    st.subheader("Answer:")
    st.write(st.session_state.last_answer["answer_with_rag"])
    st.subheader("Sources:")
    for citation in st.session_state.last_answer["citations"].split("\n"):
        st.write(citation)

# Add helpful information
st.markdown("---")

# About section with enhanced explanations
st.markdown("""
### About Anveshak

**General:** Anveshak (meaning "seeker" in Sanskrit) is a Retrieval-Augmented Generation (RAG) system that answers questions about spirituality based on insights from Indian 
spiritual texts. It searches through a database of texts to find relevant passages and generates concise answers based on those passages.

**Technical:** The RAG pipeline is powered by an underlying large embedding model that converts texts into searchable 
vector representations (using FAISS indices) and a Large Language Model (LLM) that summarizes and presents the findings of the system.

#### Goal
The path and journey to the SELF is designed to be undertaken alone. The all-encompassing knowledge is internal and not external. 
Not all find the SatGuru - the true, real Guru, for guidance.

It has been observed that all of the necessary knowledge for spiritual progress is available in Indian spiritual texts. 
Additionally, great Saints, Sages, Siddhas, Yogis, Sadhus, Rishis, Gurus, Mystics, and Spiritual Masters of all genders, backgrounds, traditions, and walks of life have imparted spiritual wisdom, 
knowledge, and guidance to beings for ages in the Indian subcontinent and beyond. Anveshak acknowledges and honors these high beings from all backgrounds equally, regardless of gender, caste, creed, or other distinctions.

Our goal is to make a small contribution to the journey of beings toward self-discovery by making this knowledge available and accessible within 
ethical, moral, and resource-based constraints. **We have no commercial or for-profit interests; this application is purely for educational purposes.** 
We sincerely hope this humble offering supports your spiritual progress. God bless.

**Additionally, through this humble effort, we offer our tribute, love, and gratitude to the higher beings — the Saints, Sages, and Masters — whose works, 
teachings, and wisdom have guided humanity and have inspired and informed this application.**

#### Important to note
- This is not a general chatbot or conversational AI. It is specifically designed to answer spiritual questions with short, concise answers based on referenced texts. It does not generate historical information, reproduce lengthy stories, create lists, or remember past conversations.
- You may receive slightly different answers when asking the same question multiple times. This variation is intentional and reflects the nuanced nature of spiritual teachings across different traditions.
- While you can select a specific number of citations and word limit, the actual response may contain fewer citations based on relevance and availability of information. Similarly, explanations may be shorter than the selected word limit if the retrieved information is concise.
- We apologize for any inconsistencies or misinterpretations that may occur. This application is educational in nature and continuously improving.
- Currently, Anveshak is only available in English.
- We do not save any user data or queries. However, user questions are processed using OpenAI's LLM service to generate responses. While we do not store this information, please be aware that interactions are processed through OpenAI's platform and are subject to their privacy policies and data handling practices.
- **Responses are generated by AI based on the retrieved texts and may not perfectly represent the original teachings, intended meaning, or context.**
- **The inclusion of any spiritual teacher, text, or tradition does not imply their endorsement of Anveshak. We reference these sources with deep respect, but claim no official affiliation.**
- **Anveshak is a tool that is not a substitute for direct spiritual guidance, personal practice, or studying original texts in their complete form. We encourage seekers to pursue deeper learning through traditional means or with the aid of experienced Spiritual Guides and Masters.**

We value your feedback to enhance this application. Please visit the *Contacts* page to share your suggestions or report any issues.

For more information about the source texts used, see *Sources* in the navigation menu.
""")

# Citation note at the bottom - improved with support message
# This encourages users to purchase original works from publishers
st.markdown('<div class="citation-note">', unsafe_allow_html=True)
st.markdown("""
The answers presented in this application are re-presented summaries of relevant passages from the listed citations. 
For the original works in their complete and authentic form, users are respectfully encouraged to purchase 
the original print or digital works from their respective publishers. Your purchase helps support these publishers 
who have brought the world closer to such important spiritual works.
""")
st.markdown('</div>', unsafe_allow_html=True)