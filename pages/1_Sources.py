import streamlit as st

# Set page config
st.set_page_config(page_title="About Sources | Anveshak: Spirituality Q&A", page_icon="📚")

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    color: #c0392b;
    text-align: center;
    margin-bottom: 1rem;
}
.section-header {
    color: #3f51b5;
    margin-top: 30px;
    margin-bottom: 15px;
    font-size: 1.8rem;
}
.subsection-header {
    color: #3f51b5;
    margin-top: 20px;
    margin-bottom: 10px;
    font-size: 1.4rem;
}
.source-container {
    background-color: #f8faff;
    border: 1px solid #e1e4f2;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
}
.featured-saints {
    background-color: #ffffff;
    border: 1px solid #e1e4f2;
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.tradition-section {
    margin-bottom: 20px;
}
</style>
<div class="main-title">About Sources</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
Anveshak draws from a rich tapestry of spiritual wisdom found in classical Indian texts, 
philosophical treatises, and the teachings of revered Saints, Sages, Siddhas, Yogis, and Spiritual Masters across centuries.
The knowledge presented here spans multiple traditions, schools of thought, and spiritual lineages
that have flourished in the Indian subcontinent and beyond.

Note that the sources consist of about 133 digitized texts, all of which were freely available over 
the internet (on sites like archive.org). Many of the texts are English translations of original 
(and in some cases, ancient) sacred and spiritual texts. All of the copyrights belong to the 
respective authors and publishers and we bow down in gratitude to their selfless work. 
Anveshak merely re-presents the ocean of spiritual knowledge and wisdom contained in the 
original works with relevant citations in a limited number of words.

See *Home* in the navigation menu for our Thank You Note for the authors and *Publishers* for acknowledgments.
""")

# Sacred Texts Section
st.markdown('<h2 class="section-header">Sacred Literature</h2>', unsafe_allow_html=True)

st.markdown("""
The foundational Indian texts consulted for Anveshak include works that have guided spiritual seekers
for centuries, providing insights into the nature of consciousness, reality, and the divine path.
""")

with st.expander("Scriptures", expanded=True):
    st.markdown("""
    - **The Vedas**: Ancient sacred texts including the Rig Veda, Sama Veda, Yajur Veda, and Atharva Veda.
    - **Upanishads**: Philosophical texts forming the theoretical basis for Hinduism.
    - **Śrīmad Bhāgavatam**: Sacred text centered on Lord Krishna and his devotees.
    - **Bhagavad Gita**: Sacred dialogue between Lord Krishna and Arjuna on the battlefield.
    - **Puranas**: Ancient texts embodying stories of the universe's creation, genealogies of kings, heroes, Sages, and demigods.
    - **Sutras**: Aphoristic texts containing essential teachings in condensed form.
    - **Dharmaśāstras**: Ancient legal texts prescribing codes of conduct, laws, ethics, and duties for individuals and society.
    - **Agamas**: Traditional texts concerning temple construction, deity worship, and philosophical doctrines.
    """)

with st.expander("Philosophical & Devotional Traditions", expanded=True):
    st.markdown("""
    - **Bhakti**: Texts and commentaries on the path of devotion.
    - **Vedanta**: Works on the philosophical system concerned with self-realization.
    - **Yoga**: Texts on the discipline of physical, mental, and spiritual practices.
    - **Patanjali**: Writings outlining the philosophy and practice of yoga.
    - **Shakti & Shaktha**: Texts focused on divine feminine energy.
    - **Tantra**: Works on esoteric traditions and practices.
    - **South Indian Vaishnavism**: Texts from the Sri Vaishnava tradition including works of Ramanuja and the Divya Prabandham.
    """)

with st.expander("Historical Epics", expanded=True):
    st.markdown("""
    - **The Mahabharata**: Ancient epic containing philosophical discourses, including the Bhagavad Gita.
    - **The Ramayana**: Epic telling the story of Lord Rama, embodying dharma and ideal conduct.
    """)

# Saints and Spiritual Masters Section
st.markdown('<h2 class="section-header">Saints and Spiritual Masters</h2>', unsafe_allow_html=True)

st.markdown("""
The teachings of numerous Saints and Spiritual Masters inform Anveshak. Their experiential 
wisdom offers profound insights into the spiritual journey.
""")

st.markdown('<div class="featured-saints">', unsafe_allow_html=True)
st.markdown("""
### Primary Teachers
The following Indian Saints and Masters constitute the core source of teachings in Anveshak (in no particular order):

Śrī Caitanya, Sri Ramakrishna, Swami Vivekananda, Sri Ramana Maharshi, Sri Nisargadatta Maharaj, 
Paramhansa Yogananda, Swami Sri Yukteswar Giri, Shri Lahiri Mahashaya, Sri Aurobindo, 
Sri Sri Ma Anandamayi, His Divine Grace A.C. Bhaktivedanta Swami Prabhupāda, 
Baba Neeb Karori (Neem Karoli Baba), Baba Lokenath, Shri Sai Baba, Bhagawan Nityananda, 
Swami Muktananda, and Swami Sivananda.

Although we genuinely wanted to have the insights from the teachings and stories of Shri Devraha Baba, but not enough information was available about the ageless Siddha Yogi.

### Other Saints
Note that while we have a strong list of Saints that influence this work, availability of 
literature and information encompassing the works of a majority of the ancient Saints are limited. 
Therefore, the teachings and works of the aforementioned primary Spiritual leaders and Saints form the core 
of Anveshak. While methods differ, the works of all of the Saints and Sages reflect a common path,
truth, and God, and therefore, Anveshak does not miss out on any important aspect of spirituality.

Comprehensive list of all of the revered Saints influencing this work (includes) follows the section *The Hidden Ones*. 
Note that Anveshak primarily relies on the works and stories of Indian Saints, Sages, Siddhas, and Yogis. However, 
we have also covered some of the revered Spiritual Masters across other traditions and regions.

Additionally, there are and there have been many other great Saints, enlightened beings, Sadhus, Sages, and Gurus who have worked 
tirelessly to uplift humanity and guide beings to their true SELF and path, of whom little is known and documented. We thank them 
and acknowledge their contribution to the world.

### The Hidden Ones
We acknowledge the boundless grace, love, compassion, and kindness, of the great Saints and Siddhas who took on a physical body 
to selflessly bless beings and guide them to the path to spirituality and the Lord. However, Saints are not required 
to come to the physical plane for continuing with their work. We would like to take a moment to thank all of those 
great beings, the enlightened ones, who visisted the physical world but remanined reclusive while working on their 
paths, and the ones who never had to take a physical bode, for their invaluable contributions and divine guidance.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Create expandable sections for different traditions
with st.expander("Advaita Vedanta & Self-Inquiry", expanded=False):
    st.markdown("""
    - Akalkot Swami
    - Appayya Dikshitar
    - Sadasiva Brahman
    - Sankara (Adi Shankaracharya)
    - Sri Nisargadatta Maharaj
    - Sri Ramakrishna
    - Sri Ramana Maharshi
    - Swami Dayananda Saraswati
    - Swami Rama Tirtha
    - Swami Sivananda
    - Swami Swayamprakasa Brahmendra Saraswati
    - Swami Vivekananda
    - Vidyaranya
    """)

with st.expander("Bhakti & Sant Tradition", expanded=False):
    st.markdown("""
    - Bhagat Pipa
    - Dadu
    - Ekanath
    - Goswami Tulsidas
    - His Divine Grace A C Bhaktivedanta Swami Prabhupada
    - Jayadeva
    - Madhva
    - Mira Bai
    - Muktabai
    - Nimbarka
    - Potana
    - Ramananda
    - Saint Haridas
    - Sakubai
    - Samartha Ramdas
    - Sant Chokhamela
    - Sant Dnyaneshwar (Jnanadev)
    - Sant Janabai
    - Sant Kabir Das
    - Sant Narahari Sonar
    - Sant Narsi Meheta
    - Sant Ravidas
    - Shri Sant Namdev Maharaj
    - Sant Tukaram Maharaj
    - Sri Caitanya (Gauranga)
    - Vallabha
    - Bhadrachalam Ramdas
    """)

with st.expander("Yoga & Kriya Yoga Lineage", expanded=False):
    st.markdown("""
    - Baba Lokenath
    - Changdev Maharaj
    - Dattatreya
    - Guru Gorakhnath
    - Guru Matsyendranath
    - Joga Paramanand
    - Paramhansa Yogananda
    - Shri Lahiri Mahashaya
    - Swami Sri Yukteswar Giri
    - Yogi Bhusanda
    - Yogi Jaigisavya
    - Yogi Mukund Rai
    """)

with st.expander("Modern Indian Saints", expanded=False):
    st.markdown("""
    - Baba Neeb Karori (Neem Karoli Baba)
    - Baba Shri Trailangaswami
    - Bal Brahmachari Maharaj
    - Bhagawan Nityananda
    - Hairakhan Baba
    - Khaki Baba
    - Narayana Guru
    - Raghavendra Swami
    - Ramalinga Swami
    - Saint Gudidi Baba
    - Shri Harshdev Puri Maharaj
    - Shri Raukhdiya Baba
    - Shri Sai Baba
    - Sombari Baba Maharaj
    - Sri Aurobindo
    - Sri Sri Ma Anandamayi
    - Sundaresa Swami
    - Swami Muktananda
    """)

with st.expander("Tamil Saints & Nayanars", expanded=False):
    st.markdown("""
    - Appar (Thirunavukkarasar)
    - Manickavasagar
    - Pattinathu Pillayar
    - Thayumanavar
    - Thiru Jnana Sambandhar
    - Thirumula Nayanar
    - Thiruvalluvar
    - Sundaramurti
    """)

with st.expander("Alvar Saints", expanded=False):
    st.markdown("""
    - Andal
    - Kulasekhara Alwar
    - Nammalwar
    - Periyalwar
    - Ramanuja
    - Saint Alavandar
    - Thirumangai Alwar
    - Thirumazhisai Alwar
    - Thiruppan Alwar
    - Thondaradippodi Alwar
    """)

with st.expander("Karnataka Saints & Haridasas", expanded=False):
    st.markdown("""
    - Gorakumbar
    - Kanak Das
    - Nilakantha Dikshitar
    - Purandara Das
    - Tyagaraja
    - Vyasaraja
    """)

with st.expander("Sikh Gurus", expanded=False):
    st.markdown("""
    - Guru Amardas
    - Guru Angad
    - Guru Arjun Singh
    - Guru Govind Singh
    - Guru Har Govind
    - Guru Har Kishan
    - Guru Har Rai
    - Guru Nanak
    - Guru Ramdas
    - Guru Tej Bahadur
    """)

with st.expander("Ancient Sages & Seers", expanded=False):
    st.markdown("""
    - Maharshi Vyasa
    - Sage Nagnath
    - Sage Riddhgiri
    - Sage Satyanath
    - Sage Yajnavalkya
    - Sant Jagamitra Naga
    - Sri Kurmadas
    """)

with st.expander("Buddhist & Jain Teachers", expanded=False):
    st.markdown("""
    - Buddha
    - Mahavira
    - Milarepa of Tibet
    - Parsvanatha
    """)

with st.expander("Sufi & Islamic Saints", expanded=False):
    st.markdown("""
    - Bulla Shah
    - Jalal-ud-din Rumi
    - Mansoor
    - Rabia
    - Shams Tabriez
    """)

with st.expander("Western & Other Traditions", expanded=False):
    st.markdown("""
    - Akha
    - Appayyacharya
    - Avadayakkal
    - Confucius
    - Damaji
    - Jesus
    - Madalasa
    - Nandanar
    - Saint Arunagiri
    - Saint Augustine
    - Saint Francis of Assisi
    - Saint Francis Xavier
    - The Saintly King Bijal
    - Vilwamangal
    - Yogi Vemana
    - Zoroaster
    """)

# Attribution note
st.markdown("""
---

### A Note on Attribution

The knowledge and wisdom contained in Anveshak comes from the teachings, writings, and oral traditions
of these Masters and texts. We have made every effort to respectfully represent their teachings 
while acknowledging that any errors in interpretation are our own.

The inclusion of a teacher or text in this list does not imply endorsement of Anveshak 
by that teacher, their lineage, or representatives. It merely indicates that their teachings 
have contributed to our understanding and have been included in our knowledge base.

We encourage users to seek out the original texts and teachings for deeper study and to support 
the publishers and organizations that continue to make these precious teachings available.
""")