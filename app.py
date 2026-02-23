import streamlit as st
import os
from dotenv import load_dotenv
from Main import process_medical_report
from Utils.RAG import RAGSystem

# Load environment variables
load_dotenv(dotenv_path='apikey.env')

st.set_page_config(
    page_title="AI Medical Diagnostics",
    page_icon="🩺",
    layout="wide",
)

def main():
    st.title("🩺 AI Agents for Medical Diagnostics")
    st.markdown("""
    Upload a medical report (text file) to get a comprehensive diagnostic analysis from our multi-agent AI system.
    You can also chat with the report to ask specific questions!
    """)

    # Custom CSS to make chat section sticky at bottom
    st.markdown("""
    <style>
    /* Make chat input sticky at bottom */
    .stChatInput {
        position: fixed;
        bottom: 3rem;
        left: 50%;
        transform: translateX(-50%);
        width: 50%;
        z-index: 1000;
    }
    /* Make web search toggle sticky on right side */
    .stCheckbox {
        position: fixed;
        bottom: 3rem;
        right: 9%;
        z-index: 1000;
        background: blue;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stCheckbox label,
    .stCheckbox span,
    .stCheckbox p,
    .stCheckbox div {
        color: white !important;
    }
    /* Make Chat with Report heading sticky on left side */
    .chat-heading {
        position: fixed;
        bottom: 3rem;
        left: 24%;
        transform: translateX(-100%);
        z-index: 1000;
        background: blue;
        padding: 0.0rem 0.5rem;
        border-radius: 5rem;
    }
    .chat-heading h3 {
        color: white !important;
        margin: 0;
        font-size: 1.2rem;
    }
    @media (max-width: 768px) {
        .stChatInput { width: 90%; bottom: 1rem; }
        .stCheckbox { bottom: 1rem; right: 5%; }
        .chat-heading { bottom: 1rem; left: 5%; transform: translateX(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # File uploader
    uploaded_file = st.file_uploader("Upload Medical Report", type=["txt"])

    if uploaded_file is not None:
        report_text = uploaded_file.read().decode("utf-8")

        # Ingest report if new file
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing report for chat..."):
                st.session_state.rag_system.ingest(report_text)
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []  # reset chat on new file

        # Show uploaded report
        with st.expander("View Uploaded Report"):
            st.text_area("Report Content", report_text, height=200)

        # Diagnostic analysis section
        st.subheader("Diagnostic Analysis")
        if "final_diagnosis" not in st.session_state:
            st.session_state.final_diagnosis = None
        if st.button("Analyze Report", type="primary"):
            with st.spinner("AI Agents are analyzing the report..."):
                try:
                    diagnosis = process_medical_report(report_text)
                    if diagnosis:
                        st.session_state.final_diagnosis = diagnosis
                        st.success("Analysis Complete!")
                    else:
                        st.error("Analysis failed to generate a result. Please check the logs.")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
        # Show diagnosis and download button
        if st.session_state.final_diagnosis:
            st.markdown(st.session_state.final_diagnosis)
            st.download_button(
                label="Download Diagnosis",
                data=st.session_state.final_diagnosis,
                file_name="diagnosis_result.txt",
                mime="text/plain",
            )

        st.markdown("---")

        # Chat with Report section (heading will be sticky)
        st.markdown('<div class="chat-heading"><h3>💬 Chat with Report</h3></div>', unsafe_allow_html=True)
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Web search toggle (sticky on right side)
        enable_web_search = st.toggle("🌐 Web Search", value=False, help="Enable web search for more comprehensive answers", key="web_search_toggle")
        # Apply toggle to RAG system
        st.session_state.rag_system.update_tools(enable_web_search)

        # Chat input (sticky at bottom)
        if prompt := st.chat_input("Ask a question about the report..."):
            # Add user message to history and display
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_system.query(prompt)
                    st.markdown(response)
            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
