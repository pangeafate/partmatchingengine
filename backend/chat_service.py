import logging
from typing import List, Dict, Any
import json
import os
import openai
from langchain.schema import Document
from vector_store import VectorStore  # Use the original version

# Set up logging
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, model: str = "gpt-4o", init_db: bool = True):
        """Initialize the chat service."""
        self.model = model
        # Set OpenAI API key from environment variable
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if not openai.api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        logger.info(f"Initializing vector store")
        
        # Check for Render environment
        if os.path.exists("/data"):
            self.vector_store = VectorStore(data_dir="/data/source", db_path="/data/chroma_db")
        else:
            self.vector_store = VectorStore()
        
        # Initialize the vector database if requested
        if init_db:
            try:
                logger.info("Initializing vector database...")
                self.vector_store.get_or_create_vector_db()
                logger.info("Vector database initialized")
            except Exception as e:
                logger.error(f"Error initializing vector database: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a context string for the LLM.
        
        This method implements an advanced context assembly strategy:
        1. Combines related chunks
        2. Includes proper citations (page numbers, section references)
        3. Sorts by relevance and adds page context
        """
        # Group documents by filename and page number
        grouped_docs = {}
        for doc in documents:
            filename = doc.metadata.get("filename", "Unknown")
            page_number = doc.metadata.get("page_number", "Unknown")
            key = f"{filename}_{page_number}"
            
            if key not in grouped_docs:
                grouped_docs[key] = []
            
            grouped_docs[key].append(doc)
        
        # Sort keys by page number for consistent presentation
        sorted_keys = sorted(grouped_docs.keys(), 
                            key=lambda k: (k.split('_')[0], 
                                          int(k.split('_')[1]) if k.split('_')[1].isdigit() else float('inf')))
        
        context_parts = []
        
        # Process each group of documents
        for key in sorted_keys:
            docs = grouped_docs[key]
            filename = docs[0].metadata.get("filename", "Unknown")
            page_number = docs[0].metadata.get("page_number", "Unknown")
            
            # Start with a header for this page
            header = f"Document: {filename} (Page {page_number})"
            page_context = f"{header}\n{'='*len(header)}\n"
            
            # Sort documents by chunk type to prioritize full pages and sections
            docs.sort(key=lambda d: {
                "page": 0,
                "section": 1,
                "page_chunk": 2
            }.get(d.metadata.get("chunk_type", ""), 3))
            
            # Extract content from each document
            for doc in docs:
                chunk_type = doc.metadata.get("chunk_type", "unknown")
                
                if chunk_type == "page":
                    # For full pages, use the content as is
                    page_context += doc.page_content + "\n"
                elif chunk_type == "section":
                    # For sections, add a section header if available
                    title_context = doc.metadata.get("title_context", "")
                    if title_context:
                        page_context += f"Section: {title_context}\n"
                    page_context += doc.page_content + "\n"
                elif chunk_type == "page_chunk":
                    # For page chunks, add a chunk indicator
                    chunk_index = doc.metadata.get("chunk_index", 0)
                    page_context += f"[Chunk {chunk_index + 1}] {doc.page_content}\n"
                else:
                    # For other document types, add as is
                    page_context += doc.page_content + "\n"
            
            context_parts.append(page_context)
        
        # Process regular JSON documents
        json_docs = [doc for doc in documents if doc.metadata.get("source") == "json_data"]
        for i, doc in enumerate(json_docs, 1):
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
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Error parsing JSON document: {e}")
                # If not valid JSON, use as is
                context_parts.append(f"Item {i}:\n{doc.page_content}\n")
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate a response to the user's query using RAG.
        
        This method implements an advanced retrieval and response generation strategy:
        1. Hybrid search combining vector similarity and keyword matching
        2. Context assembly with proper citations
        3. Detailed instructions to the LLM for accurate responses
        """
        if chat_history is None:
            chat_history = []
        
        # Ensure vector database is initialized
        if not hasattr(self.vector_store, 'vector_db') or self.vector_store.vector_db is None:
            logger.info("Vector database not initialized, initializing now...")
            try:
                self.vector_store.get_or_create_vector_db()
                logger.info("Vector database initialized")
            except Exception as e:
                logger.error(f"Error initializing vector database: {e}")
                return "I'm having trouble accessing the knowledge base. Please try again later."
        
        # Retrieve relevant documents from the vector store (increased to 10 for better coverage)
        try:
            logger.info(f"Performing similarity search for query: {query}")
            relevant_docs = self.vector_store.similarity_search(query, k=10)
            logger.info(f"Found {len(relevant_docs)} relevant documents")
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "I encountered an issue searching for information. Please try again later."
        
        # Simple re-ranking: prioritize documents with titles that match query terms
        query_terms = set(query.lower().split())
        
        def score_document(doc):
            # Base score from vector similarity (already factored in by the retrieval)
            score = 1.0
            
            # Boost score for title matches
            title_context = doc.metadata.get("title_context", "").lower()
            if title_context:
                matching_terms = sum(1 for term in query_terms if term in title_context)
                score += matching_terms * 0.5
            
            # Boost score for page-level documents
            if doc.metadata.get("chunk_type") == "page":
                score += 0.3
            elif doc.metadata.get("chunk_type") == "section":
                score += 0.2
            
            # Boost score for exact keyword matches
            page_content_lower = doc.page_content.lower()
            for term in query_terms:
                if term in page_content_lower:
                    score += 0.2
                    
                    # Give extra boost for exact product name matches
                    if len(term) > 4:  # Only boost for meaningful terms, not short words
                        # Look for exact matches surrounded by spaces or punctuation
                        if f" {term} " in page_content_lower or f"{term}." in page_content_lower:
                            score += 0.5
            
            return score
        
        # Re-rank documents
        relevant_docs.sort(key=score_document, reverse=True)
        
        # Take top 7 documents after re-ranking
        relevant_docs = relevant_docs[:7]
        
        # Format the retrieved documents into context
        try:
            context = self.format_context(relevant_docs)
            logger.info(f"Formatted context length: {len(context)}")
        except Exception as e:
            logger.error(f"Error formatting context: {e}")
            context = "Error retrieving context information."
        
        # Prepare the messages for the OpenAI API with detailed instructions
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant that answers questions about industrial parts and equipment. "
                "Your task is to provide accurate information based on the context provided. "
                "\n\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. Always include specific page numbers in your responses, e.g., 'According to page 5...'\n"
                "2. When providing part numbers, ensure they are exact matches from the documentation\n"
                "3. For questions about compatibility, cite the specific sections that mention compatibility\n"
                "4. If asked about a specific part number, look for exact matches in the context\n"
                "5. If you don't know the answer based on the provided information, say so clearly\n"
                "6. Do not make up information or part numbers\n"
                "7. When answering questions about shrink boots or connectors, provide the exact part number format as shown in the documentation\n"
                "8. Pay special attention to product names, especially trademarked names like 'MasterWrap™'\n"
                "9. Look for variations in product names and terms (e.g., with/without spaces, with/without symbols)\n"
                "10. When responding about proprietary products, include any trademarked symbols (™, ®) in your response\n"
                "11. Be attentive to specialized industry terminology and maintain exact spelling and formatting\n"
                "12. If a product has special features or unique selling points, highlight these in your response\n"
                "\n\n"
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
        try:
            logger.info(f"Sending request to OpenAI API")
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more precise answers
                max_tokens=1000
            )
            logger.info(f"Got response from OpenAI API")
            return response.choices[0].message['content']
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "I'm having trouble generating a response right now. Please try again later."