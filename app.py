import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers


# This Python script uses Streamlit to create a web app that generates a blog post based on user input.
# It defines a function get_response to interact with the LLAma 2 model for text generation.
# The user can input text, specify the number of words for the generated blog, and choose a writing style.
# The script then uses the LLAma 2 model to generate a blog post in the selected style with the specified word count.
# The generated blog post is displayed on the web app.

## Method to get response from LLAma 2

def get_response(input_text, no_words, blog_style):

    ### LLAma 2 model
    llama = ctransformers.CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                                       model_type='llama',
                                       config = {
                                           'max_new_tokens': int(no_words),
                                           'temperature': 0.01,
                                       })

    prompt_template = PromptTemplate(
        input_variables=["input_text", "no_words", "blog_style"],
        template="""
        You are a blogger. You are given the following information:
        {input_text}
        You are asked to write a blog in the style of {blog_style} that is {no_words} words long.
        Your response should be in the form of a blog post.
        """
    )

    prompt = prompt_template.format(input_text=input_text, no_words=no_words, blog_style=blog_style)

    ## generate response
    response = llama(prompt)

    return response.strip()


st.set_page_config(
    page_title="Generate Blog",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
    )

st.title("Generate Blog 💬")

input_text = st.text_input("Enter your text here:")

## creating 2 more columns

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("Enter number of words you want to generate:")

with col2:
    blog_style = st.selectbox(
        "Writing the blog in the style of:",
        ["Formal", "Informal", "Technical", "Casual"], index=0
    )

submit = st.button("Generate")

if submit:
    st.write(get_response(input_text, no_words, blog_style))
    st.balloons()




