import streamlit as st
import tiktoken # 토큰 개수를 세기위해
from loguru import logger # streamlit 구동기록을 로그로 남기기 위해

from langchain.chains import ConversationalRetrievalChain # 메모리를 가진 chain을 사용하기 위해
from langchain.chat_models import ChatOpenAI # openai llm

from langchain.document_loaders import PyPDFLoader # pdf 파일
from langchain.document_loaders import Docx2txtLoader #docx 파일
from langchain.document_loaders import UnstructuredPowerPointLoader # 여러 유형의 문서들을 넣어도 이해 가능하도록 만들기 위해

from langchain.text_splitter import RecursiveCharacterTextSplitter # text를 나누기 위해
from langchain.embeddings import HuggingFaceEmbeddings # 한국어에 특화된 embedding model을 불러오기 위해

from langchain.memory import ConversationBufferMemory # 설정한 개수만큼 대화를 메모리에 자장 하기 위해
from langchain.vectorstores import FAISS # vectorstore 저장하기 위해

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# main function
def main():
    st.set_page_config(     # streamlit 페이지의 상세 설정
    page_title="DirChat",   # 페이지 이름
    page_icon=":books:")    # 페이지 아이콘

    st.title("_Private Data :red[QA Chat]_ :books:")
    # 페이지 내 제목
    # __ : 글자 눕히기
    # :books: : 책 아이콘

    if "conversation" not in st.session_state:
        st.session_state.conversation = None     # 미리 none이라고 선언해야 코드 뒤 부분에서 오류 방지

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None     # 미리 none이라고 선언해야 코드 뒤 부분에서 오류 방지

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None  # 미리 none이라고 선언해야 코드 뒤 부분에서 오류 방지

    # sidebar 
    with st.sidebar:

        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)    # file_uploader 기능
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")                 # text_input
        process = st.button("Process")                                                                           # Process 버튼

    if process:     # Process 버튼을 누른 경우

        if not openai_api_key:          # 가장 먼저 openai_api_key 입력 여부 확인
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # 만약 openai_api_key가 입력 되었다면
        files_text = get_text(uploaded_files)       # 파일 load
        text_chunks = get_text_chunks(files_text)   # text를 chunk로 분할
        vetorestore = get_vectorstore(text_chunks)  # 벡터화
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key)  
        # get_conversation_chain : vetorestore을 통해 llm이 답변할 수 있도록 chain을 구성

        # process버튼을 누르면 업로드된 파일을 text -> 벡터화 -> api_key 확인 -> llm

        st.session_state.processComplete = True

    # 채팅화면 구현하기
    if 'messages' not in st.session_state:   # assistant 메시지 초기값으로 아래 문장을 넣어줌
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): # message 마다 with 문으로 묶어 주는 것
            st.markdown(message["content"])    # 어떤 역할에 따라 아이콘을 부여하고 content에 해당하는 문장을 적어라

    history = StreamlitChatMessageHistory(key="chat_messages")  # 이전 답변을 참고하기 위해 history 생성

    # Chat logic
    # 질의응답
    if query := st.chat_input("질문을 입력해주세요."):                         # 사용자가 질문을 입력 한다면
        st.session_state.messages.append({"role": "user", "content": query}) # 먼저 user 역할을 부여하고 content에 질문을 넣어 준다

        with st.chat_message("user"):   # user
            st.markdown(query)          # user 아이콘

        # assistant가 답변
        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):   # 로딩 중 이라는 것을 기호로 시각화 및 Thinking...을 보여줌
                result = chain({"question": query})  # chain = session_sate.conversation(), query를 llm에 넣어주면 나오는 답변을 result에 저장

                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                    # 이전에 주고 받은 답변 기록을 chat_history에 저장
                response = result['answer']   # answer를 response에 저장
                source_documents = result['source_documents']  # 참고한 문서를 source_documents에 저장

                st.markdown(response)  # assistant 아이콘 옆 적히는 컨텐츠

                with st.expander("참고 문서 확인"):  # expander : 접고 필수 있는 것, 아래는 접히는 부분에 대한 선언
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    # 첫번째에 source_documents의 metadata를 마크다운, 글+help = 데이터의 어떤 페이지 및 글을 참고했는지 마우스를 대면 나오도록 함
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # session_state에 assistant가 답변한 것도 저장함

# 토큰 개수를 기준으로 text를 split해주는 함수
def tiktoken_len(text):

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    return len(tokens)

# 업로드한 파일을 모두 text로 전환하는 함수
def get_text(docs):

    doc_list = []   # 여러 개의 파일 처리를 위해 선언
    
    for doc in docs:
        file_name = doc.name      # doc 객체의 이름을 파일 이름으로 사용, 불러온 파일이름을 file_name에 저장
        with open(file_name, "wb") as file:      # 파일을 doc.name으로 저장, file_name을 열고 
            file.write(doc.getvalue())           # 원래 doc에 있는 value들을 이 file에 적는 다는 것
            logger.info(f"Uploaded {file_name}") # file_name 업로드 한 내역을 업로드 해서 로그를 남김
            
            
        # 클라우드 상에서 다양한 파일을 처리해주기 위해
        if '.pdf' in doc.name:   # pdf 문서라면
            loader = PyPDFLoader(file_name)  
            documents = loader.load_and_split()  # 페이지별로 분할
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)  # extend를 통해서 documents 값들을 doc_list에 저장

    return doc_list

# chunk split
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

# chunk들을 벡터화
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(       # 임베딩 모델 선언
                                        model_name="jhgan/ko-sroberta-multitask", 
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}  # 벡터저장소에 저장해서 사용자의 질문과 비교하기 위해
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings) # 벡터저장소 선언

    return vectordb

# 위에서 선언한 함수들 구현을 위한 함수
def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(  
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            # 메모리를 사용하기 위해, chat_history라는 key값을 가진 chat기록을 찾아서 context에 집어넣어서 llm이 답변할 때 이전 대답을 찾아보도록 함
            # output_key='answer' : 대화를 나눈 것 중 answer에 해당하는 부분만 저장
            get_chat_history=lambda h: h,  # lambda h: h = 메모리가 들어온 그대로 get_chat_history에 저장
            return_source_documents=True,  # llm이 참고한 문서를 반환
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
