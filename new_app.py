import streamlit as st

# Create a function to display content in containers
def display_content(container_name):
    if container_name == "Tab 1":
        st.write("This is the content of Tab 1")
    elif container_name == "Tab 2":
        st.write("This is the content of Tab 2")
    elif container_name == "Tab 3":
        st.write("This is the content of Tab 3")

# Create the main tab layout
tabs = ["Tab 1", "Tab 2", "Tab 3"]
selected_tab = st.selectbox("Select a tab to hover over:", tabs)

# Create a container to display content when hovering over tabs
with st.beta_container():
    display_content(selected_tab)





