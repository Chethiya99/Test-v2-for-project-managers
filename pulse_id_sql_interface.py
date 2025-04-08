__import__('pysqlite3')
import sys
import os
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import re
import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process, LLM

# Page Configuration
st.set_page_config(
    page_title="Pulse iD - Database Query & Email Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'merchant_data' not in st.session_state:
    st.session_state.merchant_data = None
if 'raw_output' not in st.session_state:
    st.session_state.raw_output = ""
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = None
if 'email_results' not in st.session_state:
    st.session_state.email_results = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'interaction_history' not in st.session_state:
    st.session_state.interaction_history = []  # Store all interactions (queries, results, emails)
if 'selected_db' not in st.session_state:
    st.session_state.selected_db = "merchant_data_singapore.db"  # Default database
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False  # Track if the database is initialized
if 'selected_template' not in st.session_state:
    st.session_state.selected_template = "email_task_description1.txt"  # Default template
if 'trigger_rerun' not in st.session_state:
    st.session_state.trigger_rerun = False  # Track if a re-run is needed
if 'custom_template_content' not in st.session_state:
    # Initialize with default template content
    default_template_path = "email_descriptions/email_task_description1.txt"
    if os.path.exists(default_template_path):
        with open(default_template_path, 'r') as file:
            st.session_state.custom_template_content = file.read()
    else:
        st.session_state.custom_template_content = """Generate a personalized email for the merchant with the following details:
        
Merchant Name: {merchant_data['name']}
Email: {merchant_data['email']}
Business Type: {merchant_data['business_type']}
        
The email should:
1. Be professional yet friendly
2. Mention potential collaboration opportunities
3. Be around 300 words
4. Include a compelling subject line
5. Have proper HTML formatting"""

# Function to read the email task description from a text file
def read_email_task_description(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
            st.session_state.custom_template_content = content  # Update with file content
            return content
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

# Function to send email
def send_email(sender_email, sender_password, receiver_email, subject, body):
    try:
        # Create a MIMEText object with HTML content
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))  # Set the email content type to HTML

        # Connect to the SMTP server
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

# Function to store sent email data in the database
def store_sent_email(merchant_id, email, sent_time):
    try:
        conn = sqlite3.connect('sent_emails.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sent_emails (
                merchantID TEXT,
                email TEXT,
                sent_time DATETIME
            )
        ''')
        cursor.execute('''
            INSERT INTO sent_emails (merchantID, email, sent_time)
            VALUES (?, ?, ?)
        ''', (merchant_id, email, sent_time))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error storing email data: {str(e)}")
        return False

# Function to extract and clean the subject line and remove it from the email body
def extract_and_clean_subject_and_body(email_body):
    # Use regex to find the subject line (e.g., "Subject: ..." or "<html><body>Subject: ...")
    subject_match = re.search(r"(?:<html>\s*<body>\s*)?Subject:\s*(.*?)(?:<br>|</body>\s*</html>|$)", email_body, re.IGNORECASE)
    if subject_match:
        subject_line = subject_match.group(1).strip()
        # Remove the subject line from the email body
        cleaned_body = re.sub(r"(?:<html>\s*<body>\s*)?Subject:\s*.*?(?:<br>|</body>\s*</html>|$)", "", email_body, flags=re.IGNORECASE).strip()
        return subject_line, cleaned_body
    return "Exciting Partnership Opportunity with Pulse iD", email_body  # Default subject if not found

# Header Section with Title and Logo
st.image("logo.png", width=150)  # Ensure you have your logo in the working directory
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📊 PulseID Merchant Scout Agent</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #555;'>Interact with your merchant database and generate emails with ease!</h4>",
    unsafe_allow_html=True
)

# Sidebar Configuration
st.sidebar.header("Settings")

def get_api_key():
    """Function to get API Key from user input"""
    return st.sidebar.text_input("Enter Your API Key:", type="password")

# Get API Key
api_key = get_api_key()
if api_key:
    st.session_state.api_key = api_key

# Database Selection
db_options = ["merchant_data_dubai.db", "merchant_data_singapore.db"]
new_selected_db = st.sidebar.selectbox("Select Database:", db_options, index=db_options.index(st.session_state.selected_db))

# Check if the database selection has changed
if new_selected_db != st.session_state.selected_db:
    st.session_state.selected_db = new_selected_db
    st.session_state.db_initialized = False  # Reset database initialization
    st.sidebar.success(f"✅ Switched to database: {st.session_state.selected_db}")

# Model Selection
model_name = st.sidebar.selectbox("Select Model:", ["llama3-70b-8192"]) #"llama3-70b-8192"

# Email Template Selection
template_options = ["email_task_description1.txt", "email_task_description2.txt", "email_task_description3.txt"]
new_selected_template = st.sidebar.selectbox("Select Email Prompt:", template_options, index=template_options.index(st.session_state.selected_template))

# Check if template selection has changed
if new_selected_template != st.session_state.selected_template:
    st.session_state.selected_template = new_selected_template
    # Read the new template content
    description_file_path = f"email_descriptions/{st.session_state.selected_template}"
    read_email_task_description(description_file_path)
    st.sidebar.success(f"✅ Selected Prompt: {st.session_state.selected_template}")

# Initialize SQL Database and Agent
if st.session_state.selected_db and api_key and not st.session_state.db_initialized:
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=0,
            model_name=model_name,
            api_key=st.session_state.api_key
        )

        # Initialize SQLDatabase
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{st.session_state.selected_db}", sample_rows_in_table_info=3)

        # Create SQL Agent
        st.session_state.agent_executor = create_sql_agent(
            llm=llm,
            db=st.session_state.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        st.session_state.db_initialized = True  # Mark database as initialized
        st.sidebar.success("✅ Database and LLM Connected Successfully!")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Custom Template Editor at the top of the main content
st.markdown("### Email promt Editor", unsafe_allow_html=True)
st.info("Edit the email promt below. Your changes will be used when generating emails.")

# Text area for editing the template content
custom_template = st.text_area(
    "Edit Email promt:",
    value=st.session_state.custom_template_content,
    height=300,
    key="custom_template_editor"
)

# Update button for the template
if st.button("Update Template"):
    st.session_state.custom_template_content = custom_template
    st.success("✅ Template updated successfully!")

# Function to render the "Enter Query" section
def render_query_section():
    st.markdown("#### Get to know the Merchant Target List:", unsafe_allow_html=True)
    
    # Predefined questions
    predefined_questions = [
    ]
    
    # Display buttons for predefined questions
    #st.markdown("**Predefined Questions:**")
    for question in predefined_questions:
        if st.button(question, key=f"predefined_{question}"):
            st.session_state.user_query = question  # Store the question in session state
            st.session_state.trigger_rerun = True  # Trigger a re-run to process the query
    
    # Text area for user input
    user_query = st.text_area("Enter your query:", placeholder="E.g., Give first three merchant names and their emails, ratings, cuisine type and reviews.", key=f"query_{len(st.session_state.interaction_history)}", value=st.session_state.get('user_query', ''))
    
    if st.button("Run Query", key=f"run_query_{len(st.session_state.interaction_history)}"):
        if user_query:
            with st.spinner("Running query..."):
                try:
                    # Define company details and agent role
                    company_details = """
                     If possible, Please always try to give answers in a table format or point wise.
                    """

                    # Prepend company details to the user's query
                    full_query = f"{company_details}\n\nUser Query: {user_query}"

                    # Execute the query using the agent
                    result = st.session_state.agent_executor.invoke(full_query)
                    st.session_state.raw_output = result['output'] if isinstance(result, dict) else result
                    
                    # Process raw output using an extraction agent 
                    extractor_llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=st.session_state.api_key)
                    extractor_agent = Agent(
                        role="Data Extractor",
                        goal="Extract merchants, emails, reviews and anything posible from the raw output if they are only available.",
                        backstory="You are an expert in extracting structured information from text.",
                        provider="Groq",
                        llm=extractor_llm 
                    )
                    
                    extract_task = Task(
                        description=f"Extract a list of 'merchants' and their 'emails', 'reviews' from the following text:\n\n{st.session_state.raw_output}",
                        agent=extractor_agent,
                        expected_output="if available, Please return A structured list of merchant names, their associated email addresses, reviews etc extracted from the given text"
                    )
                    
                    # Crew execution for extraction 
                    extraction_crew = Crew(agents=[extractor_agent], tasks=[extract_task], process=Process.sequential)
                    extraction_results = extraction_crew.kickoff()
                    st.session_state.extraction_results = extraction_results if extraction_results else ""
                    st.session_state.merchant_data = st.session_state.extraction_results
                    
                    # Append the query and results to the interaction history
                    st.session_state.interaction_history.append({
                        "type": "query",
                        "content": {
                            "query": user_query,
                            "raw_output": st.session_state.raw_output,
                            "extraction_results": st.session_state.extraction_results
                        }
                    })
                    
                    # Trigger a re-run to update the UI
                    st.session_state.trigger_rerun = True
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query before clicking 'Run Query'.")

# Display Interaction History
if st.session_state.interaction_history:
    st.markdown("### Interaction History:", unsafe_allow_html=True)
    for idx, interaction in enumerate(st.session_state.interaction_history):
        if interaction["type"] == "query":
            st.markdown(f"#### Query: {interaction['content']['query']}")
            st.markdown("**Raw Output:**")
            st.write(interaction['content']['raw_output'])
            
            # Only display extracted merchants if there is data and it does not contain ''
            if interaction['content']['extraction_results'] and interaction['content']['extraction_results'].raw and 'errorhappened' not in interaction['content']['extraction_results'].raw:
                #st.markdown("**Extracted Merchants:**")
                #st.write(interaction['content']['extraction_results'].raw)
                
                # Show the "Generate Emails" button for this specific interaction
                if st.button(f"Generate Emails For Above Extracted Merchants", key=f"generate_emails_{idx}"):
                    with st.spinner("Generating emails..."):
                        try:
                            # Define email generation agent 
                            llm_email = LLM(model="groq/llama-3.3-70b-versatile", api_key=st.session_state.api_key)
                            email_agent = Agent(
                                role="Assume yourself as a lead Marketing Lead, with years of experiences working for leading merchant sourcing and acquiring companies such as wirecard, cardlytics, fave that has helped to connect with small to medium merchants to source an offer. Generate a personalized email for merchants with a compelling and curiosity-piquing subject line that feels authentic and human-crafted, ensuring the recipient does not perceive it as spam or automated",
                                goal="Generate personalized marketing emails for merchants.Each email should contains at least 300 words",
                                backstory="You are a marketing expert named 'Rasika Galhena' of Pulse iD fintech company skilled in crafting professional and engaging emails for merchants.",
                                verbose=True,
                                allow_delegation=False,
                                llm=llm_email 
                            )

                            # Use the custom template content from the editor
                            email_task_description = st.session_state.custom_template_content

                            # Email generation task using extracted results 
                            task = Task(
                                description=email_task_description.format(merchant_data=interaction['content']['extraction_results'].raw),
                                agent=email_agent,
                                expected_output="Marketing emails for each selected merchant, tailored to their business details. Please use a line to seperate each emails if there is more than one"
                            )

                            # Crew execution 
                            crew = Crew(agents=[email_agent], tasks=[task], process=Process.sequential)
                            email_results = crew.kickoff()
                            
                            # Display results 
                            if email_results.raw:
                                # Split the email results into individual emails (assuming emails are separated by a delimiter like "---")
                                individual_emails = email_results.raw.split("---")
                                
                                # Store each email separately in the interaction history
                                for i, email_body in enumerate(individual_emails):
                                    if email_body.strip():  # Skip empty emails
                                        # Ensure the email body is properly formatted as HTML
                                        formatted_email_body = f"""
                                        <html>
                                            <body>
                                                {email_body.replace("\n", "<br>")} 
                                            </body>
                                        </html>
                                        """
                                        
                                        # Append the generated email to the interaction history
                                        st.session_state.interaction_history.append({
                                            "type": "email",
                                            "content": formatted_email_body,
                                            "index": len(st.session_state.interaction_history)  # Unique index for each email
                                        })
                                
                                # Trigger a re-run to update the UI
                                st.session_state.trigger_rerun = True

                        except Exception as e:
                            st.error(f"Error generating emails: {str(e)}")
        
        elif interaction["type"] == "email":
            st.markdown("#### Generated Email:")
            
            # Display the full email content (including the subject line)
            st.markdown(interaction['content'], unsafe_allow_html=True)
            
            # Add a "Send" button for each email
            if st.button(f"Send Email {interaction['index'] + 1}", key=f"send_email_{interaction['index']}"):
                with st.spinner("Sending email..."):
                    try:
                        # Extract and clean the subject line and remove it from the email body
                        subject_line, cleaned_body = extract_and_clean_subject_and_body(interaction['content'])
                        
                        # Extract merchant ID and email from the interaction
                        merchant_id = interaction['content'].split("Dear ")[1].split(",")[0]  # Extract merchant name
                        receiver_email = re.findall(r'[\w\.-]+@[\w\.-]+', interaction['content'])[0]  # Extract email
                        receiver_email = 'jayan@pulseid.com'

                        # Sender email and password (replace with your credentials)
                        sender_email = "satoshinakumuto@gmail.com"
                        sender_password = "giha zfat jiqz hpbo"

                        # Send the email with the cleaned subject line and cleaned body
                        if send_email(sender_email, sender_password, receiver_email, subject_line, cleaned_body):
                            # Store the sent email data in the database
                            sent_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if store_sent_email(merchant_id, receiver_email, sent_time):
                                st.success(f"✅ Email sent to {receiver_email} and stored in the database.")
                            else:
                                st.error("Failed to store email data in the database.")
                        else:
                            st.error("Failed to send email.")
                    except Exception as e:
                        st.error(f"Error sending email: {str(e)}")
        
        st.markdown("---")

# Always render the "Ask questions about your database" section
render_query_section()

# Trigger a re-run if needed
if st.session_state.trigger_rerun:
    st.session_state.trigger_rerun = False  # Reset the trigger
    st.rerun()  # Force a re-run of the script

# Footer Section 
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Powered by <strong>Pulse iD</strong> | Built with 🐍 Python and Streamlit</div>",
    unsafe_allow_html=True 
)
