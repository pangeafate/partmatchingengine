from typing import List, Dict, Any
import json
import os
import openai
from langchain.schema import Document

from vector_store import VectorStore

class ChatService:
    def __init__(self, model: str = "gpt-4o"):
        """Initialize the chat service.
        
        Args:
            model: The OpenAI model to use
        """
        self.model = model
        # Set OpenAI API key from environment variable
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.vector_store = VectorStore()
        
        # Initialize the vector database
        self.vector_store.get_or_create_vector_db()
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Check if this is a PDF document with page information
            if doc.metadata.get("source") == "pdf_json_data":
                page_number = doc.metadata.get("page_number", "Unknown")
                filename = doc.metadata.get("filename", "Unknown")
                chunk_index = doc.metadata.get("chunk_index", None)
                
                # Format header based on whether this is a full page or a chunk
                if chunk_index is not None:
                    header = f"Document: {filename} (Page {page_number}, Chunk {chunk_index + 1})"
                else:
                    header = f"Document: {filename} (Page {page_number})"
                
                formatted_content = f"{header}\n{'='*len(header)}\n{doc.page_content}\n"
                context_parts.append(formatted_content)
            else:
                # Try to parse as JSON for other document types
                try:
                    content = json.loads(doc.page_content)
                    # Format the content in a readable way
                    formatted_content = f"Item {i}:\n"
                    formatted_content += f"ID: {content.get('id', 'N/A')}\n"
                    formatted_content += f"Name: {content.get('name', 'N/A')}\n"
                    formatted_content += f"Description: {content.get('description', 'N/A')}\n"
                    
                    # Format specifications if they exist
                    specs = content.get('specifications', {})
                    if specs:
                        formatted_content += "Specifications:\n"
                        for key, value in specs.items():
                            formatted_content += f"  - {key}: {value}\n"
                    
                    # Format compatibility if it exists
                    compatibility = content.get('compatibility', [])
                    if compatibility:
                        formatted_content += "Compatible with:\n"
                        for item in compatibility:
                            formatted_content += f"  - {item}\n"
                    
                    # Add manufacturer
                    formatted_content += f"Manufacturer: {content.get('manufacturer', 'N/A')}\n"
                    
                    context_parts.append(formatted_content)
                except (json.JSONDecodeError, TypeError):
                    # If not valid JSON, use as is
                    context_parts.append(f"Item {i}:\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate a response to the user's query using RAG."""
        if chat_history is None:
            chat_history = []
        
        # Retrieve relevant documents from the vector store
        relevant_docs = self.vector_store.similarity_search(query)
        
        # Format the retrieved documents into context
        context = self.format_context(relevant_docs)
        
        # Prepare the messages for the OpenAI API
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant that answers questions about industrial parts and equipment. "
                "Use the following information to answer the user's question. "
                "When providing information, include references to the page numbers where the information was found. "
                "For example, 'According to page 5 of the document...' or 'As mentioned on page 12...'. "
                "If you don't know the answer based on the provided information, say so. "
                "Do not make up information.\n\n"
                f"Context information:\n{context}"
            )}
        ]
        
        # Add chat history
        for message in chat_history:
            messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        # Add the current query
        messages.append({"role": "user", "content": query})
        
        # Generate a response using the OpenAI API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message['content']
