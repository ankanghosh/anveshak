import streamlit as st

# Set page config
st.set_page_config(page_title="Publishers | Anveshak: Spirituality Q&A", page_icon="📚")

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    color: #c0392b;
    text-align: center;
    margin-bottom: 1rem;
}
.publisher-container {
    background-color: #f8faff;
    border: 1px solid #e1e4f2;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
}
.section-header {
    color: #3f51b5;
    margin-top: 20px;
    margin-bottom: 15px;
}
.publisher-list {
    margin-bottom: 10px;
    line-height: 1.6;
}
</style>
<div class="main-title">Publisher Acknowledgments</div>
""", unsafe_allow_html=True)

# Introduction text
st.markdown("""
We extend our sincere gratitude to all publishers whose works have contributed to this 
educational resource. Anveshak would not be possible without their dedication 
to preserving and sharing spiritual wisdom.

The texts used in Anveshak represent a diverse collection of Indian spiritual 
literature, carefully selected to provide comprehensive insights into various spiritual 
traditions and philosophies. See list of publishers whose publications have guided us below.
""")

# List of publishers - ALPHABETICALLY SORTED
st.markdown('<h2 class="section-header">Publishers</h2>', unsafe_allow_html=True)

publishers = [
    "A Dutton Paperback - E. P. DUTTON - NEW YORK",
    "Advaita Ashrama, Kolkata (Calcutta)",
    "AHAM Publications - Asheboro, NC - U.S.A.",
    "ARKANA - PENGUIN BOOKS",
    "BELL TOWER / NEW YORK",
    "BLUE DOVE PRESS SAN DIEGO • CALIFORNIA",
    "celextel.org",
    "Chetana",
    "Clydesdale Press®",
    "Collier Books, New York - Collier Macmillan Publishers, London",
    "Compiled by the Dharmic Scriptures Team",
    "COSMO PUBLICATIONS - NEW DELHI INDIA",
    "David Godman, Avadhuta Foundation, Colorado, USA",
    "Element Books Ltd",
    "GANESH & CO., MADRAS LUZAC & CO., LONDON",
    "Gita Press, Gorakhpur, India",
    "Hampton Roads Publishing Company, Inc.",
    "HarperOne - An Imprint of HarperCollins Publishers",
    "HAY HOUSE, INC.",
    "ISKCON PRESS - Division of the International Society for Krishna Consciousness",
    "Lama Foundation",
    "Lokenath Divine Life Mission - Kolkata",
    "LONDON - GEORGE ALLEN & UNWIN LTD RUSKIN HOUSE • MUSEUM STREET",
    "Love Serve Remember Foundation",
    "Love Serve Remember Foundation or Hanuman Foundation - Santa FE, NM",
    "MOTILAL BANARSIDASS",
    "OTAM Books",
    "OTTO HARRASSOWITZ- WIESBADEN",
    "P. Ramanath Pai Press",
    "Prabhupāda Saṅkīrtana Society (England)",
    "PRABUDDHA BHARATA or AWAKENED INDIA",
    "PROBSTHAIN & CO., ORIENTAL BOOKSELLERS AND PUBLISHERS, LONDON",
    "Prof. M. RANGACHARYA MEMORIAL TRUST TRIPLICANE",
    "PUBLISHED FOR THE BOMBAY THEOSOPHICAL PUBLICATION FUND BY RAJARAM TUKARAM TATYA",
    "Rudra Press",
    "Rupa Publications India Pvt. Ltd.",
    "SELF-REALIZATION FELLOWSHIP - Los Angeles, California",
    "Shambhala Publications, Inc.",
    "SHANTI SADAN 2 9 CBBPSTOW VILLAS LONDON, W.11 1952 - Printed in Great Britain at the BURLEIGH PRESS, Lewins Mead, BRISTOL",
    "Shree Gurudev Ashram",
    "SHREE SHREE ANANDAMAYEE CHARITABLE SOCIETY - CALCUTTA",
    "SHREE SHREE ANANDAMAYEE SANGHA - KANKHAL, HARIDWAR",
    "Shree Shree Ma Anandamayee Archive®",
    "Shri Sai Baba Sansthan, Shirdi",
    "Sounds True - BOULDER, COLORADO",
    "Sri Aurobindo Ashram Publication Department",
    "Sri Ramakrishna Math, Chennai",
    "SRI KAINCHI HANUMAN MANDIR & ASHRAM - KAINCHI, DISTRICT NAINITAL",
    "SRI RAMAKRISHNA MATH",
    "Sri Ramanasramam - Tiruvannamalai",
    "Taos Music & Art, Inc.",
    "THE ACORN PRESS - Durham, North Carolina",
    "THE BHAKTIVEDANTA BOOK TRUST",
    "THE DIVINE LIFE SOCIETY",
    "The Divine Life Society Sivananda Ashram, Rishikesh, India",
    "THE PHILOSOPHICAL LIBRARY, INC. - New York, N.Y.",
    "The Theosophical Publishing House, Chennai",
    "World Wisdom, Inc.",
    "Yogi Impressions Books Pvt. Ltd.",
    "Yogoda Satsanga Society of India",
    "ZEN PUBLICATIONS - A Division of Maoli Media Private Limited"
]

# Display publishers in a simple list format
st.markdown('<div class="publisher-container">', unsafe_allow_html=True)

# Create a bulleted list of publishers
for publisher in publishers:
    st.markdown(f"- {publisher}")

st.markdown('</div>', unsafe_allow_html=True)

# Statement at the bottom
st.markdown("""
---

### About the Texts
The texts represented in Anveshak cover a wide range of spiritual traditions including 
Advaita Vedanta, Bhakti Yoga, Kashmir Shaivism, Tantra, Yoga and Kriya Yoga traditions, 
Buddhist philosophy, Jainism, Sikhism, South Indian Vaishnavism, Sufism, and the teachings 
of various Saints and Spiritual Masters from India and beyond. These texts embody centuries 
of spiritual inquiry, devotion, and wisdom.

### About Copyright
We acknowledge that all rights to the original texts remain with their respective publishers and 
copyright holders. Anveshak serves purely educational purposes and does not claim ownership 
of any of the source materials. The content is presented in summarized form with appropriate citations, 
and we have intentionally implemented word limits and other measures to respect copyright.
""")