# Let's Talk with Yourself

This repository contains the solution for the "Let's Talk with Yourself" assignment for the AT82.05: Natural Language Processing. The project demonstrates a chatbot built using Retrieval-Augmented Generation (RAG) techniques integrated with the LangChain framework.

## Project Overview

This project is divided into three main tasks:

1. **Source Discovery and Prompt Design**
   - Identification and listing of relevant personal sources (documents, resume, websites).
   - Custom design of a prompt template for the chatbot to answer questions about personal information.
   - Exploration of additional text-generation models (including OpenAI models and Groq's llama-3 3-70b-versatile).

2. **Analysis and Problem Solving**
   - Implementation of a RAG pipeline that utilizes both retriever and generator models.
   - A detailed analysis of any issues encountered (e.g., instances where models returned unrelated information) with a focus on both the retriever and generator components.

3. **Chatbot Development – Web Application**
   - A web application with a user-friendly chat interface allowing users to interact with the chatbot.
   - The chatbot provides coherent, context-aware responses and includes references to source documents supporting the answers.
   - Handling of 10 mandatory questions (e.g., "How old are you?", "What is your highest level of education?", etc.) with each question–answer pair exported in a JSON format.

## Repository Structure

- **a6_rag.ipynb**  
  The main Jupyter Notebook containing the complete implementation, including:
  - Source discovery and listing of reference documents.
  - Custom prompt design.
  - Implementation of the RAG pipeline with analysis of the models used.
  - Development of the chatbot web application.

- **app/**  
  Folder containing the code for the web application that demonstrates the chatbot.  
  (This folder includes all necessary files to run the web server and the chat interface.)

- **images/ss.png**  
  A screenshot image of the web application interface.

- **README.md**  
  This file, which provides an overview of the project, instructions for use, and details on the implementation.