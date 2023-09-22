import streamlit as st

# Define CSS styles for the side tabs and hover effect
side_tab_styles = """
    .side-tabs {
        display: flex;
        flex-direction: column;
        padding: 10px;
    }
    .side-tab {
        padding: 10px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .side-tab:hover {
        background-color: #f0f0f0;
    }
"""

# Inject the CSS styles into the Streamlit app
st.markdown(f'<style>{side_tab_styles}</style>', unsafe_allow_html=True)

# Create a dictionary mapping tab labels to their content
tab_content = {
    "Tab 1": "This is the content of Tab 1",
    "Tab 2": "This is the content of Tab 2",
    "Tab 3": "This is the content of Tab 3",
}

# Create a sidebar for the side tabs
selected_tab = st.sidebar.radio("Select a tab:", list(tab_content.keys()))

# Display the content of the selected tab
st.write(tab_content[selected_tab])
