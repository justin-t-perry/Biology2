
import os
import streamlit as st
import openai
import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize




# nltk file setup
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)  # Add custom nltk_data path

# Download the necessary data
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)




# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Define the raw experience document as a variable
document = """
Aiden S.	Quick learner in math, enjoys puzzles. Lives with grandparents. Shy in group settings.
Maria G.	Struggles with reading comprehension. Loves art and storytelling. Speaks Spanish at home.
Liam T.	Enthusiastic about science. ADHD diagnosis — needs structured tasks. Loves experiments.
Sofia R.	Very social, excels in group work. Slightly behind in math. Plays on soccer team.
Jayden L.	Tech-savvy, curious about robotics. Sometimes rushes through work. Lives with single mom.
Isabella M.	Strong in writing, especially poetry. Introverted. Volunteers at the local library.
Ethan K.	Struggles with focus. Loves video games and storytelling. Gets overwhelmed easily.
Chloe H.	Very organized and detail-oriented. Enjoys reading historical fiction. Strong in all subjects.
Noah D.	Visual learner, great at drawing. Needs help with following verbal instructions.
Emma W.	Passionate about animals. Wants to be a vet. Active in science club. Lives on a farm.
Mason J.	Hands-on learner, enjoys building things. Reading is a challenge. Works well 1-on-1.
Olivia F.	Bilingual (French/English). Loves creative writing. Sensitive to criticism.
Lucas C.	Struggles with confidence. Responds well to positive reinforcement. Loves geography.
Ava N.	Top of the class in math and science. Perfectionist tendencies. Plays violin.
Benjamin B.	Enjoys helping classmates. Sometimes misses subtle instructions. Lives with older siblings.
Charlotte Y.	Curious, loves asking questions. Struggles with staying on task. Enjoys acting.
Elijah P.	Learns best through movement and games. Enjoys gym class. Needs redirection often.
Mia Z.	Hardworking, quiet. Loves animals and reading. Lives in a multigenerational household.
Logan A.	Quick thinker but has trouble showing work. Competitive. Enjoys logic puzzles.
Amelia S.	Strong in social studies. Writes detailed journal entries. Has anxiety about tests.
James R.	Very analytical. Interested in coding. Needs help organizing his work.
Harper M.	Loves group projects. Natural leader. Needs reminders to let others speak.
Henry T.	Strong reader. Family recently moved. Needs time to adjust. Enjoys comic books.
Evelyn K.	Detail-focused, great at spelling. Doesn’t speak much in class. Loves nature walks.
Alexander V.	Gets bored easily unless challenged. Advanced in math. Plays chess competitively.
Abigail D.	Loves drama and performing. Expresses herself through art. Struggles with focus.
Sebastian N.	New to the country, English learner. Very strong in math. Kind and observant.
Ella J.	Enjoys working with her hands. Loves gardening. Struggles with abstract concepts.
Jackson E.	Athletic, kinesthetic learner. Needs help with writing. Motivated by goals.
Grace L.	Always willing to help others. Needs encouragement to challenge herself. Loves music.
Matthew Q.	Independent thinker. Loves astronomy. Can seem distracted but absorbs information deeply.


"""

# Split the document into chunks based on paragraphs or headings
def split_into_chunks(document):
    chunks = document.split("\n\n")  
    return [{"content": chunk} for chunk in chunks if chunk.strip()]  

# Initialize BM25 index
def initialize_bm25(chunks):
    tokenized_chunks = [word_tokenize(chunk["content"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, tokenized_chunks

# Retrieve relevant chunks based on user query
def retrieve_relevant_chunks(query, bm25, chunks, tokenized_chunks, k=2):
    tokenized_query = word_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i]["content"] for i in top_k_indices]

# Query OpenAI GPT with retrieved context and user query
def query_gpt(context, user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {user_input}"}
        ],
        max_tokens=200,  
        temperature=0.7  
    )
    return response['choices'][0]['message']['content'].strip()

# Streamlit 
st.title("Professional Experience Chatbot with RAG")
st.write("Ask me anything related to Justin Perry's Professional background! (For example: Tell me about Justin's AI experience, or Tell me about Justin's hobbies)")

# Process the document
chunks = split_into_chunks(document)
bm25, tokenized_chunks = initialize_bm25(chunks)

# User query input
user_input = st.text_input("Your Question:")

if user_input:
    # Retrieve relevant context
    retrieved_context = retrieve_relevant_chunks(user_input, bm25, chunks, tokenized_chunks, k=1)
    context = "\n".join(retrieved_context)

    # Query GPT
    answer = query_gpt(context, user_input)

    # Display response
    st.write("### Answer:")
    st.write(answer)
