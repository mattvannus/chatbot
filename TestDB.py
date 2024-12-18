import os
import openai
from chromadb import Client
from chromadb.config import Settings

client = Client(Settings(persist_directory=r"C:\Users\vannus8553\PycharmProjects\ChatWithMatt\chroma_db"))
print("Collections available:", client.list_collections())


