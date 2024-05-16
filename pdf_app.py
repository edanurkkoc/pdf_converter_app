import streamlit as st 
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI 
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

#Sidebar contents
with st.sidebar:
    st.title('LLM CHAT APP')
    st.markdown('''
    ## About
    Bu uygulama LLM destekli bir sohbet robotudur:
    - [Streamlit](https://streamlit.io)
    - [Langchain](https://www.langchain.com)
    - [OpenAI](https://www.langchain.com) LLM Model      
    
    ''')
    add_vertical_space(5)
    st.write('Bu uygulamanin yapimcisi benim https://github.com/edanurkkoc')
    

def main():
    st.header("CHAT WİTH PDF")
    
    load_dotenv()
    
    #upload a PDF file
    pdf=st.file_uploader("PDF'i yukle",type='pdf')
    #st.write(pdf.name)
    
    #st.write(pdf)
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks=text_splitter.split_text(text=text)
        
        # # embeddings
      
        store_name=pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectoreStore=pickle.load(f)
            #st.write('Embeddings Loaded from the Disk')
            
        else:
            embeddings=OpenAIEmbeddings()
            VectoreStore=FAISS.from_text(chunks,embeddings=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectoreStore,f)
        
        
        #Accept user question / query
        query=st.text_input("Ask questions about your PDF file:")
        #st.write(query)
        
        if query:
            docs=VectoreStore.similary_search(query=query,k=3)
            #st.write(docs)
            # #st.write('Embeddings Computation Copmleted')
            #st.write(chunks)
            llm=OpenAI(model_name='gpt-3.5-turbo')
            chain=load_qa_chain(llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,questions=query)
                print(cb)
            st.write(response)
                
                



if __name__=='__main__':
    main()