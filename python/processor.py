import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = None
qa_chain = None

def load_pdf_into_memory(text: str):
    global vectorstore, qa_chain
    print(f"Received text length: {len(text)} characters")

    if not text.strip() or len(text) < 100:
        raise ValueError("No valid text extracted from the PDF.")

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
        print(f"Created {len(docs)} document chunks")

        vectorstore = FAISS.from_documents(docs, embedding)

        llm = ChatGroq(
            temperature=0.7,
            model_name="deepseek-r1-distill-llama-70b",
            max_tokens=512
        )

        prompt_template = """<s>[INST] You are an AI assistant that summarizes documents.
        Use ONLY the following context to answer the question. Be concise (3-5 sentences).
        If the question can't be answered from the context, say so.

        Context: {context}

        Question: {question} [/INST]

        Summary:"""

        QA_PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=False
        )
        print("PDF indexing completed successfully with Groq.")
    except Exception as e:
        print(f"Error indexing PDF: {e}")
        raise

def answer_question(question: str, chat_history: list):
    if not qa_chain:
        return "❌ Please upload a PDF first."

    try:
        result = qa_chain({"query": question})
        answer = result.get('result', "No answer found in the document.")

        if "don't know" in answer.lower():
            return "I couldn't find sufficient information in the document to answer that question."
        return answer
    except Exception as e:
        return f"❌ Error answering: {str(e)}"
