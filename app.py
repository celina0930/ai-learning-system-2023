import streamlit as st
import page1, page2, page3
   
def main():
    col1, col2 = st.columns([1, 6])
    logo_image = "logo.png"
    col1.image(logo_image, width=100)
    col2.title("AI Learning System")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Analysis", "Project", "Testing"])

    if page == "Data Analysis":
        page1.data_analysis()
    elif page == "Project":
        page2.page2()
    elif page == "Testing":
        page3.page3()

if __name__ == "__main__":
    main()
