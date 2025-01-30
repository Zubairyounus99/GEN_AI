import os
import PyPDF2
import faiss
import numpy as np
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
import gdown

# Set up Groq client (replace with your actual Groq API key)
client = Groq(api_key=os.environ.get("Groq_Api_Key"))  # Use environment variable for the API key

# Paths for storing and loading processed data
DATA_FOLDER = "./pdfs"
PROCESSED_DATA_FILE = "./processed_data.pkl"

# Predefined Google Drive folder links
GOOGLE_DRIVE_FOLDER_LINKS = [
    "https://drive.google.com/drive/folders/1e5rpHuCVNxutjrnkWhsw5Dxz6gS_jfCt?usp=sharing",
    # Add more links if needed
]

# Function to download and process PDFs
def download_and_process_pdfs():
    if os.path.exists(PROCESSED_DATA_FILE):
        st.write("Pre-processed data file already exists. Skipping processing.")
        return

    st.write("Downloading and processing PDFs...")
    pdf_texts = []
    doc_names = []

    # Ensure the folder exists
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    for folder_url in GOOGLE_DRIVE_FOLDER_LINKS:
        try:
            gdown.download_folder(folder_url, quiet=False, output=DATA_FOLDER)
        except Exception as e:
            st.error(f"Error downloading folder: {e}")
            continue

    # Process each PDF
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            doc_names.append(filename)
            with open(os.path.join(DATA_FOLDER, filename), "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() for page in reader.pages)
                pdf_texts.append(text)
                st.write(f"Processed: {filename}, Text Length: {len(text)}")

    # Tokenize, chunk, and embed the data
    chunks = []
    for document in pdf_texts:
        chunks.extend(chunk_text(document))

    # Generate embeddings using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunks)

    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(X.shape[1])
    faiss_index.add(X.toarray().astype(np.float32))

    # Save processed data for future use
    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "faiss_index": faiss_index, "doc_names": doc_names}, f)

    st.write("Pre-processing completed. Data saved for future use.")

# Function to chunk the documents into smaller sections
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to retrieve relevant chunks based on a user query
def retrieve_relevant_chunks(query, faiss_index, chunks, vectorizer, k=3):
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    _, indices = faiss_index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

# Function to generate a response based on the user query and retrieved document chunks
def generate_response(query, context_chunks):
    prompt = " ".join(context_chunks) + "\n\nUser's question: " + query
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",  # Use the required model
        stream=False,
    )
    return chat_completion.choices[0].message.content

# Streamlit app
def main():
    st.title("Chatbot for UEP Procedures")
    st.write("This app uses pre-processed data to instantly answer your questions.")

    # Load pre-processed data
    if not os.path.exists(PROCESSED_DATA_FILE):
        st.error("Pre-processed data file is missing. Please rebuild the app.")
        return

    with open(PROCESSED_DATA_FILE, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]
    vectorizer = data["vectorizer"]
    faiss_index = data["faiss_index"]
    doc_names = data["doc_names"]

    # Sidebar with document names
    st.sidebar.title("Documents List")
    st.sidebar.markdown("<style>ul {font-family: 'Arial', sans-serif; font-size: 1.1rem;}</style>", unsafe_allow_html=True)
    st.sidebar.markdown("<ul>" + "".join(f"<li>{name}</li>" for name in doc_names) + "</ul>", unsafe_allow_html=True)

    # Chat interface
    st.subheader("Ask Your Question")
    conversation_history = []

    user_query = st.text_input("Your question:")
    if st.button("Submit"):  # Submit button for querying
        if user_query:
            context_chunks = retrieve_relevant_chunks(user_query, faiss_index, chunks, vectorizer)
            response = generate_response(user_query, context_chunks)
            st.write(f"**Response:** {response}")

            # Save conversation history
            conversation_history.append({"question": user_query, "response": response})
        else:
            st.warning("Please enter a question.")

    # Option to view conversation history
    if st.checkbox("Show Conversation History"):
        for entry in conversation_history:
            st.write(f"**Question:** {entry['question']}")
            st.write(f"**Response:** {entry['response']}")

if __name__ == "__main__":
    # Ensure the PDFs are processed at build time
    download_and_process_pdfs()
    main()
