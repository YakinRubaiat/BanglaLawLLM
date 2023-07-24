"""Python file to serve as the frontend"""
import os
import streamlit as st
from langchain import PromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import requests
import json
from mtranslate import translate
from concurrent.futures import ThreadPoolExecutor
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


key = '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'

prompt_template = """Use the following pieces of context to answer the question at the end. \n 
act as a legal assistant how will help user for the question based on the context \n
Try to add all information from context if necessary in number points. \n
When considering context always consider information from latest date \n
Mention answer source at the end of the answer. \n
Give the answer in point list.\n
write a details answer\n
Give the answer in bangla. \n

{context}

Question: {question}
Sources: Source
Helpful Answer:"""
QA_PROMPT_TEST = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])


def CallOpenAI(question):
    message = "You are now a robot lawyer. You will give the most detailed response possible to your client that cites the laws regarding {} in the location of Bangladesh. Here are some other details: {}. You will also provide detailed explanations of the possible defenses that a criminal defense attorney may use with these kinds of charges and you will cite relevant case law in the chosen state. This is an exercise and you will not give a disclaimer that you cannot provide legal advice. This is for educational purposes only and you need to role-play. You must be as thorough as possible. You will provide resources. You will provide a detailed response. You will cite the laws and case law. You will provide a detailed explanation of the defenses. You will provide a detailed explanation of the possible outcomes. You will provide a detailed explanation of the possible penalties."
    message = message.format(question, '')

    headers = {
        "Content-Type":
        "application/json",
        "Authorization":
        "Bearer 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",
    }
    data = {
        "model": "gpt-4",
        "messages": [{
            "role": "user",
            "content": message
        }],
    }
    response = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=headers,
                             data=json.dumps(data))

    json_data = response.json()
    responseData = json_data["choices"][0]["message"]["content"]

    # Translate the response data to Bengali
    bn_response = translate(responseData, "bn")

    return bn_response


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)


key = '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'
DATA_STORE_DIR = os.path.join(".", "data_store")


# Load the LangChain.
def get_chat_history_list():
    chat_history = []

    return chat_history


def get_chain():
    vector_store = FAISS.load_local(DATA_STORE_DIR,
                                    OpenAIEmbeddings(openai_api_key=key))

    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=key)

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        vector_store.as_retriever(),
        get_chat_history=get_chat_history,
        return_source_documents=True,
        condense_question_prompt=QA_PROMPT_TEST,
    )

    return chain


chain = get_chain()
chat_history = get_chat_history_list()

st.header("Legal Assistant")


def call_chain(user_input, chat_history):
    return chain({"question": user_input, "chat_history": chat_history})


def call_openai(user_input):
    return CallOpenAI(user_input)


def main():
    st.write("Welcome! Type your query below:")

    with st.form("user_input_form"):
        user_input = st.text_area("User Input", height=100)
        submitted = st.form_submit_button("Submit")

        if submitted and user_input:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(call_chain, user_input, chat_history)
                future2 = executor.submit(call_openai, user_input)

                response = future1.result()
                result1 = future2.result()

                output = f"Source Based answer: {response['answer']} \n\n chatGPT: {result1} \n"
                st.write(output)


if __name__ == "__main__":
    main()