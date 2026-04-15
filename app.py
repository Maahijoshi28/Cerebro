import streamlit as st
import os
import io
from groq import Groq # Added for voice transcription
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage


from dotenv import load_dotenv
# --- STEP 1: PAGE CONFIG ---
st.set_page_config(page_title="Cerebro Voice & Text", page_icon="🧠", layout="wide")

# --- STEP 2: BACKEND CONFIG (Secure for Render) ---
# This loads your local .env file only if you are testing on your laptop
load_dotenv()

# These lines grab the keys you pasted into the Render Dashboard
GROQ_KEY = os.getenv("GROQ_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# Safety Check: If keys are missing, the app will show a clear message instead of crashing
if not GROQ_KEY or not TAVILY_KEY:
    st.error("❌ API Keys Missing! Please ensure GROQ_API_KEY and TAVILY_API_KEY are set in Render's Environment Variables.")
    st.stop()

# Set environment variables for LangChain/Tavily tools to use internally
os.environ["GROQ_API_KEY"] = GROQ_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_KEY

# Initialize the Groq client
client = Groq(api_key=GROQ_KEY)

# --- STEP 3: UI STYLING ---
st.markdown("""
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #ffffff; }
    .main-title { color: #1f2937; font-size: 4rem; font-weight: 800; text-align: center; margin-top: -60px; }
    .expert-tag { color: #2563eb; font-size: 1.2rem; text-align: center; font-weight: 600; margin-bottom: 20px; }
    .stButton>button { background: linear-gradient(90deg, #2563eb, #1d4ed8); color: white; border-radius: 8px; font-weight: bold; width: 100%; height: 3.5em;}
    </style>
    """, unsafe_allow_html=True)

# --- STEP 4: SIDEBAR ---
with st.sidebar:
    st.markdown("### 🏛️ Research Hub")
    domain = st.selectbox("Choose Field:", ["General Knowledge", "Medical & Healthcare", "Academic & Thesis", "Financial & Market Analysis", "Legal & Regulatory", "Tech & Software Trends"])
    st.info(f"Mode: **{domain}**")
    st.caption("Developed by Maahi Joshi")

# --- STEP 5: MULTI-AGENT ENGINE ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    research_data: str
    report_draft: str
    critique: str
    iteration: int

def researcher_node(state: AgentState):
    search_tool = TavilySearchResults(max_results=3)
    query = f"{domain}: {state['messages'][0].content}"
    results = search_tool.invoke(query)
    return {"research_data": str(results), "iteration": state.get('iteration', 0) + 1}

def editor_node(state: AgentState):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
    prompt = f"Expert Persona: {domain}. Using this data: {state['research_data']}, write a formal report on {state['messages'][0].content}."
    response = llm.invoke(prompt)
    return {"report_draft": response.content}

def fact_checker_node(state: AgentState):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    prompt = f"Verify this {domain} report: {state['report_draft']}. APPROVED or Critique."
    return {"critique": llm.invoke(prompt).content}

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node); workflow.add_node("editor", editor_node); workflow.add_node("fact_checker", fact_checker_node)
workflow.add_edge(START, "researcher"); workflow.add_edge("researcher", "editor"); workflow.add_edge("editor", "fact_checker")
workflow.add_conditional_edges("fact_checker", lambda x: END if "APPROVED" in x["critique"] or x["iteration"] >= 2 else "researcher")
app_engine = workflow.compile()

# --- STEP 6: UI DISPLAY (Voice + Text Input) ---
st.markdown('<p class="main-title">Cerebro</p>', unsafe_allow_html=True)
st.markdown(f'<p class="expert-tag">Domain: {domain}</p>', unsafe_allow_html=True)

# Container for input
input_container = st.container()
with input_container:
    # A cleaner layout: Voice input first
    voice_file = st.audio_input("🎤 Click to Speak your Research Topic")
    
    # Transcription logic
    spoken_text = ""
    if voice_file:
        try:
            with st.spinner("Transcribing..."):
                # Pass the audio file to Groq Whisper
                transcription = client.audio.transcriptions.create(
                    file=("sample.wav", voice_file.read()),
                    model="whisper-large-v3",
                )
                spoken_text = transcription.text
                st.success(f"Recognized: {spoken_text}")
        except Exception as e:
            st.error(f"Voice Error: {e}")

    # Text Input (populated by spoken_text if available)
    topic = st.text_input("Enter Topic or verify spoken text:", value=spoken_text, placeholder="e.g., Future of AI in Chandigarh")

# --- STEP 7: LAUNCH ---
if st.button("🚀 INITIATE MULTI-AGENT PANEL"):
    if not topic:
        st.warning("Please provide a topic via voice or text.")
    else:
        try:
            with st.status(f"⚡ {domain} Experts Collaborating...", expanded=True) as status:
                state = {"messages": [HumanMessage(content=topic)], "iteration": 0}
                for output in app_engine.stream(state):
                    for node_name, _ in output.items():
                        st.write(f"✔️ **{node_name.capitalize()} Agent** finished.")
                
                final_state = app_engine.invoke(state)
                status.update(label="Complete!", state="complete")
            
            st.info(final_state["report_draft"])
            st.download_button("📥 Download Report", final_state["report_draft"], file_name="Report.txt")
        except Exception as e:
            st.error(f"Error: {e}")
