from setuptools import find_packages, setup

setup(
    name = 'MEDI_PDF_CHATBOT',
    version= '0.0.1',
    author= 'Saran Raj',
    author_email= 'Mr.saranraj777@gmail.com',
    packages= find_packages(),
    install_requires = ["langchain","langchain-openai","langchain-huggingface","qdrant-client","langchain-qdrant","langchain-community","pypdf","python-dotenv","IPython","streamlit","sentence-transformers"]

)


