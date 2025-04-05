import streamlit as st

# Set page configuration
st.set_page_config(page_title="Contact Us | Anveshak: Spirituality Q&A", page_icon="🕉️")

# Custom CSS for styling
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    color: #c0392b;
    text-align: center;
    margin-bottom: 1rem;
}
.contact-container {
    background-color: #f8f5ff;
    border: 1px solid #e0d6fe;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.email-highlight {
    font-weight: bold;
    color: #6a1b9a;
}
.contact-header {
    color: #6a1b9a;
    font-size: 1.5rem;
    margin-bottom: 15px;
}
</style>
<div class="main-title">Contact Us</div>
""", unsafe_allow_html=True)

# Main content
st.markdown('<div class="contact-container">', unsafe_allow_html=True)
st.markdown('<div class="contact-header">We Value Your Feedback</div>', unsafe_allow_html=True)
st.markdown("""
Hello!

We are grateful for your interest in Anveshak. 
This journey of spiritual exploration is one we walk together, and your insights are invaluable to us.

If you have:
- **Questions or clarifications** about the spiritual concepts presented
- **Suggestions for improvement** to enhance the learning experience
- **Technical issues** you've encountered while using Anveshak
- **Recommendations for additional spiritual texts** that could enrich our knowledge base
- **Personal insights or reflections** you wish to share from your spiritual journey

Please reach out to us at <span class="email-highlight">ankanatwork@gmail.com</span>

Your feedback helps us refine this humble drop of knowledge and make spiritual wisdom more accessible to seekers everywhere. 
It is often said that knowledge grows when shared, and wisdom deepens through communion.

We hope Anveshak serves as a meaningful companion on your spiritual path.

With gratitude and reverence,

The Anveshak Team
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Additional information
st.markdown("""
### Our Commitment

We are committed to preserving the authenticity and sanctity of the spiritual teachings presented in Anveshak. 
We see ourselves not as creators of knowledge, but as caretakers and facilitators of timeless spiritual truth, values, 
and wisdom passed down through generations.

Your feedback helps us honor this responsibility with greater care and precision. We strive to continually improve based 
on your insights and experiences.

Thank you for being part of this sacred journey of learning and discovery.
""")