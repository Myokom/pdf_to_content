import streamlit as st
import os
import tempfile
import io
import random
import openai
import re


from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
import ast

st.set_page_config(
    page_title="PDF to Content",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed")


# ---------------- Helpers ----------------

def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError) as e:
        st.error(f"Error: The provided input is not correctly formatted {e}")
        st.stop()

def get_randomized_options(options):
    correct_answer = options[0] 
    random.shuffle(options)
    return options, correct_answer

def extract_text_from_pdf(file_obj):
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "temp_pdf.pdf")

    # Write the content of the file object to the temporary file
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(file_obj.read())

    # Initialize the PyPDFLoader with the temporary file path
    pdf_loader = PyPDFLoader(temp_file_path)
    document_list = pdf_loader.load_and_split()  # This returns a list of Document objects

    # Extract text from each Document object
    combined_text = " ".join([doc.page_content for doc in document_list])

    # Optionally, delete the temporary file and directory here if you want to clean up
    os.remove(temp_file_path)
    os.rmdir(temp_dir)

    return combined_text

def get_quiz_data(text, openai_api_key):
    quiz_template = f"""
    You are a helpful assistant programmed to generate questions based on any text provided. For every chunk of text you receive, you're tasked with designing 5 distinct questions. Each of these questions will be accompanied by 3 possible answers: one correct answer and two incorrect ones. 

    For clarity and ease of processing, structure your response in a way that emulates a Python list of lists. 

    Your output should be shaped as follows:

    1. An outer list that contains 5 inner lists.
    2. Each inner list represents a set of question and answers, and contains exactly 4 strings in this order:
    - The generated question.
    - The correct answer.
    - The first incorrect answer.
    - The second incorrect answer.

    Your output should mirror this structure:
    [
        ["Generated Question 1", "Correct Answer 1", "Incorrect Answer 1.1", "Incorrect Answer 1.2"],
        ["Generated Question 2", "Correct Answer 2", "Incorrect Answer 2.1", "Incorrect Answer 2.2"],
        ...
    ]

    It is crucial that you adhere to this format as it's optimized for further Python processing. Don't include anything like ```python to indicate it's Python code

    """
    try:
        system_message_prompt = SystemMessagePromptTemplate.from_template(quiz_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY),
            prompt=chat_prompt,
        )
        quiz_data = chain.run(extracted_text)
    
        quiz_data_clean = ast.literal_eval(quiz_data)
        return chain.run(text)
    
    except Exception as e:
        if "AuthenticationError" in str(e):
            st.error("Incorrect API key provided. Please check and update your API key.")
            st.stop()
        elif 'context_length_exceeded' in str(e):
            token_count = re.search(r'your messages resulted in (\d+) tokens', str(e)).group(1)
            st.error(f"The uploaded file is too large with {token_count} tokens. "
                     "This application uses GPT-3.5-turbo with a maximum token limit of 16,385. "
                     f"Your file exceeds this limit by {int(token_count) - 16385} tokens. "
                     "Please reduce the file size to within 16,385 tokens and try uploading again.")
            st.stop()
        else:
            st.error(f"An error occurred: {str(e)}")
            st.stop()


def get_flash_data(text, openai_api_key):
    flash_template = f"""
    You are an assistant programmed to generate flashcards based on any text provided. Your task is to identify the 10 most important terms from the provided text and create flashcards for each term with its definition.

    For clarity and ease of processing, structure your response in a way that emulates a Python list of lists. 

    Your output should be shaped as follows:

    1. An outer list that contains 10 inner lists.
    2. Each inner list represents a flashcard and contains exactly 2 strings in this order:
    - The term.
    - The definition.

    Your output should mirror this structure:
    [
        ["Term 1", "Definition of Term 1"],
        ["Term 2", "Definition of Term 2"],
        ...
        ["Term 10", "Definition of Term 10"]
    ]

    It is crucial that you adhere to this format as it's optimized for further Python processing. Don't include anything like ```python to indicate it's Python code

"""
    try:
        system_message_prompt = SystemMessagePromptTemplate.from_template(flash_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY),
            prompt=chat_prompt,
        )
        flashcards_data = chain.run(extracted_text)
    
        flashcards_data_clean = ast.literal_eval(flashcards_data)
        #st.code(flashcards_data_clean)  # Display the cleaned data
        return flashcards_data_clean
    
    except openai.BadRequestError as e:
        if 'context_length_exceeded' in str(e):
            token_count = re.search(r'resulted in (\d+) tokens', str(e)).group(1)
            st.error(f"The uploaded file is too large with {token_count} tokens. "
                     "This application uses GPT-3.5-turbo with a maximum token limit of 16,385. "
                     f"Your file exceeds this limit by {int(token_count) - 16385} tokens. "
                     "Please reduce the file size to within 16,385 tokens and try uploading again.")
            st.stop()
        else:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

    except SyntaxError as se:
        st.error(f"A syntax error occurred when processing the data: {str(se)}")
        st.text("Raw data causing syntax error:")
        st.code(flashcards_data)  # Display the raw data causing the error
        st.stop()

    except ValueError as ve:
        st.error(f"A value error occurred when processing the data: {str(ve)}")
        st.text("Raw data causing value error:")
        st.code(flashcards_data)  # Display the raw data causing the error
        st.stop()

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.text("Raw data causing unexpected error:")
        st.code(flashcards_data)  # Display the raw data causing the error
        st.stop()

# Function to summarize text
def summarize_text(text, OPENAI_API_KEY):
    try:
        system_message_prompt = SystemMessagePromptTemplate.from_template("Summarize the following text:")
        human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        chain = LLMChain(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY),
            prompt=chat_prompt,
        )
        
        summary = chain.run(text)
        return summary

    except openai.BadRequestError as e:
        if 'context_length_exceeded' in str(e):
            token_count = re.search(r'your messages resulted in (\d+) tokens', str(e)).group(1)
            st.error(f"The uploaded file is too large with {token_count} tokens. "
                     "This application uses GPT-3.5-turbo with a maximum token limit of 16,385. "
                     f"Your file exceeds this limit by {int(token_count) - 16385} tokens. "
                     "Please reduce the file size to within 16,385 tokens and try uploading again.")
            st.stop()
        else:
            st.error(f"An error occurred: {str(e)}")
            st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()

# ---------------- Main page ----------------

# Create form for user input
with st.form("user_input"):
    OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key:", placeholder="sk-XXXX", type='password')

    # Initialize session state for uploaded files and extracted text if not already present
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = None
    if 'extracted_text' not in st.session_state:
        st.session_state['extracted_text'] = ""

    # File uploader widget
    uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=False)

    # When the user uploads a file, process it and save the content to the session state
    if uploaded_files is not None:
        with st.spinner('Processing PDF...'):
            # Extract text from the uploaded PDF and store it in session state
            extracted_text = extract_text_from_pdf(uploaded_files)
            st.session_state['extracted_text'] = extracted_text

    submitted = st.form_submit_button("Craft my content!")


tab1, tab2, tab3 = st.tabs(["Summary", "Multiple Choice", "Flashcards"])


# ---------------- Data ----------------


with tab1:
    if 'summary' not in st.session_state:
        if 'extracted_text' in st.session_state and OPENAI_API_KEY:
            with st.spinner('Wait for it... summarizing your text! 🙄'):
                # Use the previously extracted text
                st.session_state['summary'] = summarize_text(st.session_state['extracted_text'], OPENAI_API_KEY)
        else:
            # Provide instructions to the user if the prerequisites are not met
            if not st.session_state.get('extracted_text'):
                st.info("Please upload a PDF document and click 'Craft my content!' to proceed.")
            elif not OPENAI_API_KEY:
                st.error("Please fill out the OpenAI API Key to proceed. If you don't have one, you can obtain it [here](https://platform.openai.com/account/api-keys).")

    # Display summary if it exists in session state
    if 'summary' in st.session_state:
        st.subheader("Summary:")
        st.markdown(st.session_state['summary'])







# ---------------- MCQ ----------------
if 'quiz_submitted' not in st.session_state:
    st.session_state['quiz_submitted'] = False
if 'balloons_shown' not in st.session_state:
    st.session_state['balloons_shown'] = False


with tab2:
    # Check if the necessary data is available to create a quiz
    if 'extracted_text' in st.session_state and st.session_state['extracted_text'] and OPENAI_API_KEY:
        with st.spinner("Crafting your quiz...🤓"):
            # Generate quiz data from the stored extracted text only if needed
            if submitted or ('quiz_data_list' not in st.session_state):
                quiz_data_str = get_quiz_data(st.session_state['extracted_text'], OPENAI_API_KEY)
                st.session_state.quiz_data_list = string_to_list(quiz_data_str)

            # Initialize or update session state variables for the quiz
            if 'user_answers' not in st.session_state:
                st.session_state.user_answers = [None for _ in st.session_state.quiz_data_list]
            if 'correct_answers' not in st.session_state:
                st.session_state.correct_answers = []
            if 'randomized_options' not in st.session_state:
                st.session_state.randomized_options = []
            
            # Generate randomized options for each question
            for q in st.session_state.quiz_data_list:
                options, correct_answer = get_randomized_options(q[1:])
                st.session_state.randomized_options.append(options)
                st.session_state.correct_answers.append(correct_answer)

            with st.form(key='quiz_form'):
                st.subheader("🧠 Test Your Knowledge!", anchor=False)
                for i, q in enumerate(st.session_state.quiz_data_list):
                    options = st.session_state.randomized_options[i]
                    default_index = st.session_state.user_answers[i] if st.session_state.user_answers[i] is not None else 0
                    response = st.radio(q[0], options, index=default_index)
                    user_choice_index = options.index(response)
                    st.session_state.user_answers[i] = user_choice_index  # Update the stored answer right after fetching it

                results_submitted = st.form_submit_button(label='Submit')

                if results_submitted:
                    # Set a flag that the quiz has been submitted
                    st.session_state['quiz_submitted'] = True
                    score = sum([ua == st.session_state.randomized_options[i].index(ca) for i, (ua, ca) in enumerate(zip(st.session_state.user_answers, st.session_state.correct_answers))])
                    st.success(f"Your score: {score}/{len(st.session_state.quiz_data_list)}")

                    # Check if all answers are correct and balloons have not been shown yet
                    if score == len(st.session_state.quiz_data_list) and not st.session_state.get('balloons_shown', False):
                        st.balloons()
                        st.session_state['balloons_shown'] = True  # Set the flag to True after showing balloons
                    elif score != len(st.session_state.quiz_data_list):
                        # Reset the flag if the user scores less upon retaking the quiz
                        st.session_state['balloons_shown'] = False
                        incorrect_count = len(st.session_state.quiz_data_list) - score
                        if incorrect_count == 1:
                            st.warning(f"Almost perfect! You got 1 question wrong. Let's review it:")
                        else:
                            st.warning(f"Almost there! You got {incorrect_count} questions wrong. Let's review them:")

                    for i, (ua, ca, q, ro) in enumerate(zip(st.session_state.user_answers, st.session_state.correct_answers, st.session_state.quiz_data_list, st.session_state.randomized_options)):
                        with st.expander(f"Question {i + 1}", expanded=False):
                            if ro[ua] != ca:
                                st.info(f"Question: {q[0]}")
                                st.error(f"Your answer: {ro[ua]}")
                                st.success(f"Correct answer: {ca}")

                # Handle the display of results when navigating back to the tab
                elif 'quiz_submitted' in st.session_state and st.session_state['quiz_submitted']:
                    # If the form was submitted at least once, continue to display the score and correct/incorrect answers
                    # without re-submitting the form
                    score = sum([ua == st.session_state.randomized_options[i].index(ca) for i, (ua, ca) in enumerate(zip(st.session_state.user_answers, st.session_state.correct_answers))])
                    st.success(f"Your score: {score}/{len(st.session_state.quiz_data_list)}")
                    for i, (ua, ca, q, ro) in enumerate(zip(st.session_state.user_answers, st.session_state.correct_answers, st.session_state.quiz_data_list, st.session_state.randomized_options)):
                        with st.expander(f"Question {i + 1}", expanded=False):
                            if ro[ua] != ca:
                                st.info(f"Question: {q[0]}")
                                st.error(f"Your answer: {ro[ua]}")
                                st.success(f"Correct answer: {ca}")


    else:
        # Provide instructions to the user if the prerequisites are not met
        if not st.session_state.get('extracted_text'):
            st.info("Please upload a PDF document and click 'Craft my content!' to proceed.")
        elif not OPENAI_API_KEY:
            st.error("Please fill out the OpenAI API Key to proceed. If you don't have one, you can obtain it [here](https://platform.openai.com/account/api-keys).")




# ---------------- Flashcards ----------------



# Initialize the flashcards in the session state if not already present
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = []

with tab3:
    # Check if the necessary data is available to create flashcards
    if 'extracted_text' in st.session_state and st.session_state['extracted_text'] and OPENAI_API_KEY:
        if submitted or ('flashcards' not in st.session_state or not st.session_state.flashcards):
            with st.spinner("Creating your flashcards... 🧠"):
                flashcards_data = get_flash_data(st.session_state['extracted_text'], OPENAI_API_KEY)
                st.session_state.flashcards = flashcards_data

        no = len(st.session_state.flashcards)
        st.caption(f"{no} flashcards have been generated!")


        # Callback function to show the answer
        def show_answer():
            st.session_state.show_answer = True

        # Callback function to go to the next flashcard
        def next_card():
            # Increment the index for the next card if not at the end
            if st.session_state.current_index < len(st.session_state.flashcards) - 1:
                st.session_state.current_index += 1
            # Show the completion message if the 'Next' button is clicked on the last flashcard
            elif st.session_state.current_index == len(st.session_state.flashcards) - 1:
                st.session_state.completed = True
            st.session_state.show_answer = False

        # Callback function to go to the previous flashcard
        def prev_card():
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.session_state.show_answer = False
                # Reset the completion flag if going back from the end
                if st.session_state.current_index < len(st.session_state.flashcards) - 1:
                    st.session_state.completed = False

        # Initialize session state variables
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
            st.session_state.show_answer = False
            st.session_state.completed = False

        # Display the current flashcard term
        term, definition = st.session_state.flashcards[st.session_state.current_index]
        st.subheader(term)

        # Column layout for buttons
        col1, col2 = st.columns(2)
        with col1:
            prev_disabled = st.session_state.current_index == 0
            prev = st.button("Previous", on_click=prev_card, disabled=prev_disabled, key="Previous", use_container_width=True)

        with col2:
            next = st.button("Next", on_click=next_card, key="Next", use_container_width=True)

        # Button to show the answer for the current flashcard
        if st.button("Show answer", on_click=show_answer, key="Answer", use_container_width=True):
            st.session_state.show_answer = True

        # Check the state flag and display the answer if the flag is set
        if st.session_state.show_answer:
            st.write(definition)

        # Show completion message if the completed flag is set
        if st.session_state.completed:
            st.success("You've completed all the flashcards!")

    else:
        # Provide instructions to the user if the prerequisites are not met
        if not st.session_state.get('extracted_text'):
            st.info("Please upload a PDF document and click 'Craft my content!' to proceed.")
        elif not OPENAI_API_KEY:
            st.error("Please fill out the OpenAI API Key to proceed. If you don't have one, you can obtain it [here](https://platform.openai.com/account/api-keys).")



