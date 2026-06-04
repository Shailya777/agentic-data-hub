import os
import sys
import re
import chromadb
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
    raw_chunks= re.split(r'(?=\n[1-9]\.0\s+)', full_text)

    # 4: Chunk Validation and Cleanup:
    clean_chunks= []
    for chunk in raw_chunks:
        cleaned= chunk.strip()
        if len(cleaned) > 50: # Filtering Out Microscopic Chunks (Blank Pages, Isolated Metadata)
            clean_chunks.append(cleaned)

    hub_logger.info(f"Successfully Parsed into {len(clean_chunks)} Semantic Policy Chunks!!")
    return clean_chunks

if __name__ == '__main__':
    pdf_path= 'C:\\Users\\shail\\PycharmProjects\\PythonProject\\agentic-data-hub\\data\\documents\\Olist_Corporate_Policies.pdf'
    if not os.path.exists(pdf_path):
        hub_logger.info(f'PDF File Not Found: {pdf_path}')
    else:
        test_chunks= extract_and_chunk_policies(pdf_path= pdf_path)
        for chunk in test_chunks:
            print(chunk)
            print('\n\n')
