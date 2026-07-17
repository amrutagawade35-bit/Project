import streamlit as st
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Page configuration
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)


# Load Model
@st.cache_resource
def load_spam_model():
    return load_model("spam_model.keras", compile=False)

model = load_spam_model()


# Load Tokenizer
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()


# ---------------- CSS ----------------

st.markdown("""
<style>

.stApp{
    background:#eef6ff;
}


/* Main box */
.main-box{
    background:#dbeafe;
    padding:30px;
    border-radius:20px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.1);
}


/* Button */
.stButton>button{

    background:#2563eb;
    color:white;

    height:55px;
    width:220px;

    border-radius:12px;

    font-size:22px;
    font-weight:bold;

    border:none;

}

.stButton>button:hover{

    background:#1d4ed8;
    color:white;

}


/* Sidebar */

[data-testid="stSidebar"]{

    background:#f8fbff;

}

</style>

""", unsafe_allow_html=True)



# ---------------- Sidebar ----------------

with st.sidebar:

    st.title("📧 Project Info")

    st.write("---")

    st.markdown("""
    **Project Name**

    Email Spam Detection System


    **Technology Used**

    🧠 Deep Learning  
    📝 NLP  
    🐍 Python  
    💻 Streamlit  
    🔥 TensorFlow


    **Model**

    Neural Network


    **Text Processing**

    Tokenizer + Padding


    **Purpose**

    Detect whether an email is Spam or Safe.
    """)



# ---------------- Main Content ----------------


st.markdown("""
<div class="main-box">

<h1 style="text-align:center;color:#1e40af;">
📧 Email Spam Detection
</h1>

<p style="text-align:center;font-size:20px;">
Deep Learning Based Spam Email Classifier
</p>

</div>

""", unsafe_allow_html=True)



st.write("")


email = st.text_area(
    "✉ Enter Email",
    height=200,
    placeholder="Type your email here..."
)



if st.button("🚀 Predict"):


    if email.strip()=="":
        st.warning("Please enter email text")


    else:

        email_seq = tokenizer.texts_to_sequences([email])


        email_pad = pad_sequences(
            email_seq,
            maxlen=100,
            padding="post",
            truncating="post"
        )


        prediction = model.predict(
            email_pad,
            verbose=0
        )


        probability = float(prediction[0][0])


        if probability >=0.5:

            st.error("🚨 SPAM EMAIL")

            st.write(
                f"Spam Confidence : {probability*100:.2f}%"
            )


        else:

            st.success("✅ SAFE EMAIL")

            st.write(
                f"Safe Confidence : {(1-probability)*100:.2f}%"
            )