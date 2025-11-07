import streamlit as st
from src.utils.insights import configure_groq

def show_chatbot_page():
    """
    Renders the context-aware chatbot page.
    """
    st.header("🤖 AI Chatbot")
    st.write("Ask general questions, or questions about your uploaded data and model results!")

    # Configure Groq client
    client = configure_groq()
    if client is None:
        st.error("Chatbot is unavailable: Could not connect to Groq.")
        return

    # Initialize chat history (done in app.py, but good to check)
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about your data, results, or anything else..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # --- THIS IS THE NEW LOGIC ---
                    
                    # 1. Create the context-aware system prompt
                    context_messages = [
                        {"role": "system", "content": "You are a helpful AI data scientist. Answer the user's questions. If the user asks about their data or models, use the context provided below. If no context is provided, just chat normally."}
                    ]

                    # 2. Add Data Context if it exists
                    if 'data' in st.session_state and st.session_state['data'] is not None:
                        data_summary = st.session_state['data'].describe().to_string()
                        context_messages.append({
                            "role": "system",
                            "content": f"CONTEXT: Here is a summary of the user's uploaded data (from df.describe()):\n{data_summary}"
                        })

                    # 3. Add Model Results Context if it exists
                    if 'model_results' in st.session_state and st.session_state['model_results'] is not None:
                        results_summary = st.session_state['model_results'].to_string()
                        context_messages.append({
                            "role": "system",
                            "content": f"CONTEXT: Here are the user's model training results:\n{results_summary}"
                        })

                    # 4. Combine context with chat history
                    # We send the system context + the entire chat history
                    messages_to_send = context_messages + st.session_state.messages
                    
                    # 5. Call Groq with the combined messages
                    chat_completion = client.chat.completions.create(
                        messages=messages_to_send,
                        model="llama-3.1-8b-instant",
                    )
                    response = chat_completion.choices[0].message.content
                    st.markdown(response)
                    
                    # 6. Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.error(f"Error communicating with Groq: {e}")