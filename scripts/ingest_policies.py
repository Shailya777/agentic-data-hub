import os
import sys
import re
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from pydantic import BaseModel, Field
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

# Updating Sys Path to add Project Root folder:;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.logger import hub_logger

# Initializing OpenAI Client:
openai= OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Pydantic Schema for LLM Summarization Step:
class ChunkMetadata(BaseModel):
    """
    Pydantic Schema for LLM Summarization Step
    """
    header: str = Field(
        description= "A short, descriptive title for this specific text chunk."
    )
    summary: str= Field(
        description= "A concise 1-2 sentence summary of the core rules in this chunk."
    )

def extract_and_chunk_policies(pdf_path: str) -> list[str]:
    """
    Reads the PDF, cleans formatting artifacts, and Semantically Chunks the text
    by splitting strictly at corporate section headers (1.0, 2.0, etc.).
    :param pdf_path: Path to the PDF file.
    :return: List of semantic chunks.
    """

    hub_logger.info(f'Extracting and Semantically Chunking PDF: {pdf_path}')

    # 1: Extracting Full Text fro PDF:
    full_text= ""
    with open(pdf_path, 'rb') as file:
        reader= PyPDF2.PdfReader(file)
        for pages in reader.pages:
            full_text += pages.extract_text() + '\n'

    # 2: Cleaning up PDF artifacts (stripping out repeating page footers):
    full_text= re.sub(r'Olist Internal Document - Strictly Confidential - Page \d+', '', full_text)

    # 3: Semantic Header Splitting:
    # Regex Lookahead: (?=\n[1-9]\.0\s+)
    # This searches for a newline, followed by a digit (1-9), a period, a zero, and a space.
    # Because it is a "lookahead", it splits the text right BEFORE the header,
    # ensuring the "1.0 Title" stays attached to its respective paragraph.
    raw_chunks= re.split(r'(?=\n[1-9]\.0\s+|\nPolicy ID:)', full_text)

    # 4: Chunk Validation, Cleanup and Inserting Policy Title on Each Chunk:
    clean_chunks= []
    current_policy_title= 'Olist_Corporate_Policy'

    for chunk in raw_chunks:
        cleaned= chunk.strip()

        if len(cleaned) < 50: # Filtering Out Microscopic Chunks (Blank Pages, Isolated Metadata)
            continue

        # Checking if Current Chunk contains a New Policy Title:
        title_match= re.search(r'Title:\s*(.+)', cleaned) # regex search to extract the text right after "Title"
        if title_match:
            current_policy_title= title_match.group(1).strip()
            clean_chunks.append(cleaned)
        else:
            # if it's a numbered chunk without Title, adding parent title at the top:
            enriched_chunk= f"Parent Document: {current_policy_title}\n\n{cleaned}"
            clean_chunks.append(enriched_chunk)

    hub_logger.info(f"Successfully Parsed into {len(clean_chunks)} Semantic Policy Chunks!!")
    return clean_chunks

def process_chunks_with_llm(chunk_text: str) -> ChunkMetadata:
    """
    Passes the semantic chunk to GPT-4o to generate a summary and header.
    :param chunk_text: Chunk text.
    :return: ChunkMetadata Object with Header and Summary for Chunk.
    """

    system_prompt= """
    You are an expert Data Librarian. Read the following corporate policy text 
    and generate a clear header and a brief summary. This metadata will be used 
    to improve retrieval accuracy in a Vector Database.
    """
    response= openai.chat.completions.parse(
        model= 'gpt-4o',
        messages= [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': chunk_text}
        ],
        response_format= ChunkMetadata,
        temperature= 0.0
    )

    return response.choices[0].message.parsed

def ingest_to_chroma(chunks: list[str]):
    """
    Embeds the chunks and stores them in an isolated ChromaDB collection.
    :param chunks: Processed Chunks from extract_and_chunk_policies with Summary and Header from process_chunks_with_llm.
    """

    hub_logger.info('Initializing ChromaDB Client...')

    # Connecting to ChromaDB Directory:
    DB_PATH= os.path.abspath(os.path.join(os.path.dirname(__file__),"../data/vector_db"))
    chroma_client= chromadb.PersistentClient(path= DB_PATH)

    # Embedding:
    embedding= embedding_functions.OpenAIEmbeddingFunction(
        api_key= os.getenv('OPENAI_API_KEY'),
        model_name= 'text-embedding-3-small'
    )

    # Wiping the Collection if Already Exists:
    try:
        chroma_client.delete_collection(name= 'olist_corporate_policies')
        hub_logger.info(f"Successfully Deleted Olist Corporate Policies Collection!")
    except ValueError:
        pass

    # Creating Collection for Company Policies in ChromaDB:
    collection= chroma_client.get_or_create_collection(
        name= 'olist_corporate_policies',
        embedding_function= embedding
    )

    for i, chunk in enumerate(chunks):
        hub_logger.info(f"Processing Chunk {i+1}/{len(chunks)} through LLM...")

        metadata= process_chunks_with_llm(chunk)

        # Constructing Chunk with Metadata:
        enhanced_document= f"Title: {metadata.header}\nSummary: {metadata.summary}\n\nContent: \n{chunk}"

        # Generating Embedding for The Chunk:
        response= openai.embeddings.create(
            input= enhanced_document,
            model= 'text-embedding-3-small'
        )
        embedding= response.data[0].embedding

        # Inserting into ChromaDB:
        collection.add(
            ids= [f"policy_chunk_{i}"],
            embeddings= [embedding],
            documents= [enhanced_document],
            metadatas= [{'source': 'Olist_Corporate_Policy.pdf', "header": metadata.header}],
        )

    hub_logger.info(f"Ingestion Complete! Corporate Policies Collection is fully populated!")

if __name__ == '__main__':
    pdf_path= 'C:\\Users\\shail\\PycharmProjects\\PythonProject\\agentic-data-hub\\data\\documents\\Olist_Corporate_Policies.pdf'
    if not os.path.exists(pdf_path):
        hub_logger.info(f'PDF File Not Found: {pdf_path}')
    else:
        chunks= extract_and_chunk_policies(pdf_path= pdf_path)
        ingest_to_chroma(chunks= chunks)
