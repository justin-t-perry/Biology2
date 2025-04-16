
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
This is 6th grade.

This is the class:

Aiden S. is a Quick learner in math, enjoys puzzles. Lives with grandparents. Shy in group settings.
Maria G. Struggles with reading comprehension. Loves art and storytelling. Speaks Spanish at home.
Liam T.	is Enthusiastic about science. ADHD diagnosis — needs structured tasks. Loves experiments.
Sofia R. is Very social, excels in group work. Slightly behind in math. Plays on soccer team.
Jayden L. is Tech-savvy, curious about robotics. Sometimes rushes through work. Lives with single mom.
Isabella M.	is Strong in writing, especially poetry. Introverted. Volunteers at the local library.
Ethan K. Struggles with focus. Loves video games and storytelling. Gets overwhelmed easily.
Chloe H. is Very organized and detail-oriented. Enjoys reading historical fiction. Strong in all subjects.
Noah D.	is a Visual learner, great at drawing. Needs help with following verbal instructions.
Emma W.	is Passionate about animals. Wants to be a vet. Active in science club. Lives on a farm.
Mason J. is a Hands-on learner, enjoys building things. Reading is a challenge. Works well 1-on-1.
Olivia F. is Bilingual (French/English). Loves creative writing. Sensitive to criticism.
Lucas C. Struggles with confidence. Responds well to positive reinforcement. Loves geography.
Ava N. is Top of the class in math and science. Perfectionist tendencies. Plays violin.
Benjamin B.	Enjoys helping classmates. Sometimes misses subtle instructions. Lives with older siblings.
Charlotte Y. is Curious, loves asking questions. Struggles with staying on task. Enjoys acting.
Elijah P. Learns best through movement and games. Enjoys gym class. Needs redirection often.
Mia Z.	Hardworking, quiet. Loves animals and reading. Lives in a multigenerational household.
Logan A. is a Quick thinker but has trouble showing work. Competitive. Enjoys logic puzzles.
Amelia S. is Strong in social studies. Writes detailed journal entries. Has anxiety about tests.
James R. is Very analytical. Interested in coding. Needs help organizing his work.
Harper M. Loves group projects. Natural leader. Needs reminders to let others speak.
Henry T. is a Strong reader. Family recently moved. Needs time to adjust. Enjoys comic books.
Evelyn K. is Detail-focused, great at spelling. Doesn’t speak much in class. Loves nature walks.
Alexander V. Gets bored easily unless challenged. Advanced in math. Plays chess competitively.
Abigail D. Loves drama and performing. Expresses herself through art. Struggles with focus.
Sebastian N. is New to the country, English learner. Very strong in math. Kind and observant.
Ella J.	Enjoys working with her hands. Loves gardening. Struggles with abstract concepts.
Jackson E. is Athletic, kinesthetic learner. Needs help with writing. Motivated by goals.
Grace L. is Always willing to help others. Needs encouragement to challenge herself. Loves music.
Matthew Q. is an Independent thinker. Loves astronomy. Can seem distracted but absorbs information deeply.





Grade 6 Science Curriculum Plan
--------------------------------
Timeframe: 6 Weeks
Theme: Earth & Life Science Integration

Week 1: The Scientific Method
- Topics: Observation, hypothesis, experimentation, conclusion
- Key Terms: Variable, control, data, analysis
- Activities:
    • Design a simple experiment (e.g., which paper towel absorbs more water?)
    • Group worksheet: Identify parts of an experiment
- Assessment:
    • Short quiz
    • Student-designed experiment presentation

Week 2: The Water Cycle
- Topics: Evaporation, condensation, precipitation, collection
- Key Terms: Transpiration, groundwater, runoff
- Activities:
    • Diagram labeling
    • Interactive simulation of the water cycle
- Assessment:
    • Build a mini “cloud in a jar”
    • Exit ticket question: “Where does rain come from?”

Week 3: Weather & Climate
- Topics: Weather tools, types of clouds, climate zones
- Key Terms: Barometer, anemometer, cumulus, climate change
- Activities:
    • Weather journal: record daily temperature and conditions
    • Build a simple barometer
- Assessment:
    • Group poster: Compare two climate zones
    • Vocabulary matching game

Week 4: Ecosystems & Food Webs
- Topics: Producers, consumers, decomposers, food chains vs. food webs
- Key Terms: Herbivore, carnivore, omnivore, trophic level
- Activities:
    • Create a local ecosystem map
    • Analyze a desert or ocean food web
- Assessment:
    • Quiz: Identify roles in a food web
    • Mini project: Design your own ecosystem

Week 5: Adaptations & Natural Selection
- Topics: Physical vs. behavioral adaptations, survival traits
- Key Terms: Mutation, camouflage, mimicry, evolution
- Activities:
    • Simulation game: Natural selection in action (e.g., colored moths on tree bark)
    • Case study: How animals adapt to Arctic climate
- Assessment:
    • Reflection: “If I were an animal, how would I survive in the desert?”
    • Partner quiz

Week 6: Human Impact on the Environment
- Topics: Pollution, recycling, climate change, conservation
- Key Terms: Carbon footprint, renewable energy, sustainability
- Activities:
    • Audit of school trash/recycling
    • Design a "green school" poster
- Assessment:
    • Presentation: Ways to reduce your impact
    • Creative writing: A day in the life of a future Earth

Chunking Tip (for RAG ingestion):
- Each week can be divided into chunks:
    • Summary
    • Vocabulary
    • Activities
    • Assessments
- This structure helps with targeted retrieval when students or teachers ask questions like:
    • “What activities support kinesthetic learners?”
    • “Find a science lesson involving climate change.”
    • “What can I use to explain natural selection visually?”



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
st.write("Ask anything related to your class! (Suggestion: Get started by asking who's in my class?)")

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
