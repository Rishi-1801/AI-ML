import validators,streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_unstructured import UnstructuredLoader
from dotenv import load_dotenv
load_dotenv()



# Streamlit setup
st.title("Langchain: Summarize Text from YT or Website")
st.subheader('Summarize URL')

# Get groq api key
with st.sidebar:
    api_key=st.text_input("Groq API Key",value="",type="password")

url=st.text_input("URL",label_visibility='collapsed')

model=ChatGroq(api_key=api_key,model="llama-3.3-70b-versatile")

template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt=PromptTemplate(
    template=template,
    input_variables=['text']
)
 

if st.button("Summarize"):
    ##validate all inputs
    if not api_key.strip() or not url.strip():
        st.error("Please provide the Information")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # load the data
                if "youtube.com" in url:
                    loader=YoutubeLoader.from_youtube_url(url)
                else:
                    loader=UnstructuredLoader(urls=[url],ssl_verfify=False)
                # Docs
                docs=loader.load()

                # Chain
                chain=load_summarize_chain(llm=model,chain_type='stuff',prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")

