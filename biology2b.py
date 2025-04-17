
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

# Define the classroom info as a variable
document = """
This is 6th grade.
the first 15 students are 11 years old, the last 15 students are 12 years old.

This is the class:

Aiden S. is a Quick learner in math, enjoys puzzles. Lives with grandparents. Shy in group settings. Loves playing chess. Enjoys hiking with his grandfather. Dreams of becoming an engineer. Age 11.

Maria G. Struggles with reading comprehension. Loves art and storytelling. Speaks Spanish at home. Enjoys cooking traditional family recipes. Loves painting murals at community events. Has a pet parrot named Paco. Age 12.

Liam T. is Enthusiastic about science. ADHD diagnosis — needs structured tasks. Loves experiments. Loves collecting rocks and minerals. Plays drums in a garage band with friends. Wants to be a marine biologist. Age 11.

Sofia R. is Very social, excels in group work. Slightly behind in math. Plays on soccer team. Organizes charity bake sales with her friends. Loves fashion design sketches. Has a twin brother. Age 11.

Jayden L. is Tech-savvy, curious about robotics. Sometimes rushes through work. Lives with single mom. Loves building computer games in Scratch. Big fan of superhero comics. Wants to invent his own robot someday. Age 12.

Isabella M. is Strong in writing, especially poetry. Introverted. Volunteers at the local library. Plays classical piano. Writes fantasy short stories in her free time. Loves visiting old bookstores. Age 11.

Ethan K. Struggles with focus. Loves video games and storytelling. Gets overwhelmed easily. Excellent at drawing video game characters. Loves building LEGO models. Takes care of a pet rabbit named Max. Age 12.

Chloe H. is Very organized and detail-oriented. Enjoys reading historical fiction. Strong in all subjects. Loves making scrapbooks. Enjoys strategy board games. Avid fan of mysteries like Nancy Drew. Age 11.

Noah D. is a Visual learner, great at drawing. Needs help with following verbal instructions. Designs his own comic books. Loves skateboarding. Wants to work as an animator. Age 11.

Emma W. is Passionate about animals. Wants to be a vet. Active in science club. Lives on a farm. Shows goats at the county fair. Rides horses competitively. Loves birdwatching. Age 12.

Mason J. is a Hands-on learner, enjoys building things. Reading is a challenge. Works well 1-on-1. Loves tinkering with old radios. Enjoys fishing with his uncle. Wants to build a treehouse this summer. Age 11.

Olivia F. is Bilingual (French/English). Loves creative writing. Sensitive to criticism. Enjoys baking French pastries. Collects vintage postcards. Loves performing poetry at open mic nights. Age 11.

Lucas C. Struggles with confidence. Responds well to positive reinforcement. Loves geography. Builds intricate maps by hand. Loves stargazing and astronomy. Keeps a travel journal for places he wants to visit. Age 12.

Ava N. is Top of the class in math and science. Perfectionist tendencies. Plays violin. Leads the school science club. Enjoys knitting as a relaxing hobby. Loves participating in math competitions. Age 11.

Benjamin B. Enjoys helping classmates. Sometimes misses subtle instructions. Lives with older siblings. Plays basketball after school. Collects trading cards. Loves cooking breakfast for his family on weekends. Age 12.

Charlotte Y. is Curious, loves asking questions. Struggles with staying on task. Enjoys acting. Writes her own short plays. Loves visiting museums. Can recite the entire script of The Lion King. Age 11.

Elijah P. Learns best through movement and games. Enjoys gym class. Needs redirection often. Loves rollerblading. Enjoys making funny home videos. Wants to be a stunt performer someday. Age 12.

Mia Z. is Hardworking, quiet. Loves animals and reading. Lives in a multigenerational household. Enjoys baking cookies with her grandmother. Keeps a journal about her pets. Loves origami art. Age 11.

Logan A. is a Quick thinker but has trouble showing work. Competitive. Enjoys logic puzzles. Plays competitive chess online. Loves detective novels. Aims to beat the school’s Rubik’s cube record. Age 11.

Amelia S. is Strong in social studies. Writes detailed journal entries. Has anxiety about tests. Loves genealogy and tracing family history. Enjoys historical reenactments. Collects old coins. Age 12.

James R. is Very analytical. Interested in coding. Needs help organizing his work. Enjoys building Raspberry Pi projects. Plays the clarinet in the school band. Volunteers to help set up school tech equipment. Age 11.

Harper M. Loves group projects. Natural leader. Needs reminders to let others speak. Is captain of the debate team. Enjoys community theater. Writes her own blog about school life. Age 11.

Henry T. is a Strong reader. Family recently moved. Needs time to adjust. Enjoys comic books. Enjoys biking new trails around his neighborhood. Big fan of superhero movies. Keeps a list of dream vacations. Age 12.

Evelyn K. is Detail-focused, great at spelling. Doesn’t speak much in class. Loves nature walks. Enjoys birdwatching. Keeps a pressed flower collection. Writes nature poems in her spare time. Age 11.

Alexander V. Gets bored easily unless challenged. Advanced in math. Plays chess competitively. Builds his own board games. Loves participating in hackathons. Wants to major in physics one day. Age 12.

Abigail D. Loves drama and performing. Expresses herself through art. Struggles with focus. Enjoys designing costumes. Dreams of being on Broadway. Loves creative makeup tutorials. Age 11.

Sebastian N. is New to the country, English learner. Very strong in math. Kind and observant. Enjoys learning new languages. Loves cooking family recipes with his mom. Dreams of traveling the world. Age 11.

Ella J. Enjoys working with her hands. Loves gardening. Struggles with abstract concepts. Has her own vegetable garden. Loves woodworking with her grandfather. Wants to open a plant nursery someday. Age 12.

Jackson E. is Athletic, kinesthetic learner. Needs help with writing. Motivated by goals. Plays soccer and basketball. Enjoys obstacle courses and ninja training gyms. Wants to be a personal trainer. Age 11.

Grace L. is Always willing to help others. Needs encouragement to challenge herself. Loves music. Plays flute in the school orchestra. Loves baking and decorating cakes. Volunteers at the local animal shelter. Age 12.

Matthew Q. is an Independent thinker. Loves astronomy. Can seem distracted but absorbs information deeply. Builds telescopes from kits. Enjoys sketching star constellations. Wants to work for NASA someday. Age 11.









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
            {"role": "system", "content": "You are a helpful assistant, acting as a teacher's assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {user_input}"}
        ],
        max_tokens=200,  
        temperature=0.7  
    )
    return response['choices'][0]['message']['content'].strip()

# Streamlit 
st.title("Virtual Teacher's Assistant with RAG")
st.write("Ask anything related to your class! (Suggestion: Get started by asking 'who's in my class'?)")

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
