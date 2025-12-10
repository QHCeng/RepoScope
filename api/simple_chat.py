import logging
import os
import re
from typing import List, Optional
from urllib.parse import unquote

from adalflow.core.types import ModelType
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, OPENAI_API_KEY
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.rag import RAG
from api.prompts import (
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT
)

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Simple Chat API",
    description="Simplified API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Models for the API
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatCompletionRequest(BaseModel):
    """
    Model for requesting a chat completion.
    """
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(None, description="Optional path to a file in the repository to include in the prompt")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    type: Optional[str] = Field("github", description="Type of repository (e.g., 'github', 'gitlab', 'bitbucket')")

    # model parameters
    provider: str = Field("google", description="Model provider (google, openai, openrouter, ollama, bedrock, azure, dashscope)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")

    language: Optional[str] = Field("en", description="Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude from processing")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude from processing")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")

@app.post("/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """Stream a chat completion response directly using Google Generative AI"""
    try:
        # Check if request contains very large input
        input_too_large = False
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                tokens = count_tokens(last_message.content)
                logger.info(f"Request size: {tokens} tokens")
                if tokens > 8000:
                    logger.warning(f"Request exceeds recommended token limit ({tokens} > 7500)")
                    input_too_large = True

        # Create a new RAG instance for this request
        try:
            request_rag = RAG(provider=request.provider, model=request.model)

            # Extract custom file filter parameters if provided
            excluded_dirs = None
            excluded_files = None
            included_dirs = None
            included_files = None

            if request.excluded_dirs:
                excluded_dirs = [unquote(dir_path) for dir_path in request.excluded_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom excluded directories: {excluded_dirs}")
            if request.excluded_files:
                excluded_files = [unquote(file_pattern) for file_pattern in request.excluded_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom excluded files: {excluded_files}")
            if request.included_dirs:
                included_dirs = [unquote(dir_path) for dir_path in request.included_dirs.split('\n') if dir_path.strip()]
                logger.info(f"Using custom included directories: {included_dirs}")
            if request.included_files:
                included_files = [unquote(file_pattern) for file_pattern in request.included_files.split('\n') if file_pattern.strip()]
                logger.info(f"Using custom included files: {included_files}")

            request_rag.prepare_retriever(request.repo_url, request.type, request.token, excluded_dirs, excluded_files, included_dirs, included_files)
            logger.info(f"Retriever prepared for {request.repo_url}")
        except ValueError as e:
            if "No valid documents with embeddings found" in str(e):
                logger.error(f"No valid embeddings found: {str(e)}")
                # Check if this is a new repository that hasn't been processed yet
                error_detail = (
                    "This repository is being processed for the first time. "
                    "This may take a few minutes depending on the repository size. "
                    "Please wait a moment and try again. "
                    "If the error persists, the repository may be empty or there may be an issue with the embedding API."
                )
                raise HTTPException(status_code=500, detail=error_detail)
            else:
                logger.error(f"ValueError preparing retriever: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error preparing retriever: {str(e)}")
        except Exception as e:
            logger.error(f"Error preparing retriever: {str(e)}")
            # Check for specific embedding-related errors
            if "All embeddings should be of the same size" in str(e):
                raise HTTPException(status_code=500, detail="Inconsistent embedding sizes detected. Some documents may have failed to embed properly. Please try again.")
            else:
                raise HTTPException(status_code=500, detail=f"Error preparing retriever: {str(e)}")

        # Validate request
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages provided")

        last_message = request.messages[-1]
        if last_message.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from the user")

        # Process previous messages to build conversation history
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]

                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(
                        user_query=user_msg.content,
                        assistant_response=assistant_msg.content
                    )

        # Check if this is a Deep Research request
        is_deep_research = False
        research_iteration = 1

        # Process messages to detect Deep Research requests
        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                # Only remove the tag from the last message
                if msg == request.messages[-1]:
                    # Remove the Deep Research tag
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()

        # Count research iterations if this is a Deep Research request
        if is_deep_research:
            research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
            logger.info(f"Deep Research request detected - iteration {research_iteration}")

            # Check if this is a continuation request
            if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
                # Find the original topic from the first user message
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"Found original research topic: {original_topic}")
                        break

                if original_topic:
                    # Replace the continuation message with the original topic
                    last_message.content = original_topic
                    logger.info(f"Using original topic for research: {original_topic}")

        # Get the query from the last message
        query = last_message.content

        # Check if this is a file listing query
        query_lower = query.lower()
        file_listing_patterns = [
            r'所有文件|complete.*file|all.*file|list.*file|文件列表|file.*list',
            r'有哪些文件|what.*file|which.*file|show.*file',
            r'完整的.*文件|complete.*list|full.*list'
        ]
        is_file_listing_query = any(re.search(pattern, query_lower) for pattern in file_listing_patterns)

        # Extract file path from conversation history if current query doesn't have one
        context_file_path = None
        file_path_patterns = [
            r'([a-zA-Z0-9_\-./]+\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))',
            r'([^\s的]+(?:/[^\s的]+)*\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))'
        ]
        
        # Check conversation history for recently mentioned files
        memory_turns = request_rag.memory()
        for turn_id, turn in reversed(list(memory_turns.items())):
            if hasattr(turn, 'user_query') and hasattr(turn.user_query, 'query_str'):
                user_query_text = turn.user_query.query_str
                for pattern in file_path_patterns:
                    match = re.search(pattern, user_query_text, re.IGNORECASE)
                    if match:
                        potential_path = match.group(1) if match.groups() else match.group(0)
                        if potential_path:
                            potential_path = potential_path.strip('"\'`的').strip()
                            if '/' in potential_path or potential_path.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.h', '.hpp', '.go', '.rs', '.html', '.css', '.md', '.txt', '.json', '.yaml', '.yml')):
                                context_file_path = potential_path
                                logger.info(f"Found file path from conversation history: {context_file_path}")
                                break
                if context_file_path:
                    break
        
        # Check if this is a file content request (e.g., "给我api/api.py的代码")
        file_content_request = None
        file_path_patterns_extended = [
            # Chinese patterns: "给我xxx的代码", "显示xxx的代码", "xxx的代码"
            r'给我\s*([^\s的]+(?:/[^\s的]+)*\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))\s*的代码',
            r'显示\s*([^\s的]+(?:/[^\s的]+)*\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))\s*的代码',
            r'([^\s的]+(?:/[^\s的]+)*\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))\s*的代码',
            # Chinese patterns: "xxx文件里", "xxx文件中", "xxx文件"
            r'([^\s的]+(?:/[^\s的]+)*\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))\s*文件里',
            r'([^\s的]+(?:/[^\s的]+)*\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))\s*文件中',
            r'([^\s的]+(?:/[^\s的]+)*\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))\s*文件',
            # English patterns: "give me code for xxx", "show code of xxx"
            r'give\s+me\s+(?:the\s+)?code\s+(?:for|of)\s+([^\s]+\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))',
            r'show\s+(?:me\s+)?(?:the\s+)?code\s+(?:for|of)\s+([^\s]+\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))',
            # English patterns: "what classes in xxx", "classes in xxx file"
            r'(?:what|which|list|show).*?(?:class|function|method|variable).*?(?:in|of)\s+([^\s]+\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))',
            r'([^\s]+\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))\s*(?:file)?\s*(?:has|contains|includes).*?(?:class|function|method)',
            # Direct file path patterns
            r'([a-zA-Z0-9_\-./]+\.(?:py|js|ts|java|cpp|h|hpp|go|rs|html|css|md|txt|json|yaml|yml))'
        ]
        
        # Try to extract file path from current query
        for pattern in file_path_patterns_extended:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Get the first non-empty group
                potential_path = next((g for g in match.groups() if g), None)
                if potential_path:
                    # Clean up the path (remove quotes, extra spaces, Chinese characters)
                    potential_path = potential_path.strip('"\'`的').strip()
                    # Check if it looks like a file path
                    if '/' in potential_path or potential_path.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.h', '.hpp', '.go', '.rs', '.html', '.css', '.md', '.txt', '.json', '.yaml', '.yml')):
                        file_content_request = potential_path
                        logger.info(f"Detected file content request for: {file_content_request}")
                        break
        
        # If no file path in current query but found in history, use it
        if not file_content_request and context_file_path:
            # Check if query is asking about "it", "that", "里面", "这个文件" etc.
            reference_patterns = [
                r'里面|这个|那个|它|it|that|this|there|里面.*?|都是哪些|有哪些|what.*?are|which.*?are|list.*?all'
            ]
            is_reference_query = any(re.search(pattern, query_lower) for pattern in reference_patterns)
            
            if is_reference_query:
                file_content_request = context_file_path
                logger.info(f"Using file path from conversation history for reference query: {file_content_request}")

        # Only retrieve documents if input is not too large
        context_text = ""
        retrieved_documents = None
        file_content_from_local = None

        if not input_too_large:
            try:
                # If it's a file content request, try to read the file from local repository
                if file_content_request:
                    try:
                        # Get local repository path from db_manager
                        if hasattr(request_rag, 'db_manager') and request_rag.db_manager.repo_paths:
                            local_repo_path = request_rag.db_manager.repo_paths.get("save_repo_dir")
                            if local_repo_path and os.path.exists(local_repo_path):
                                file_full_path = os.path.join(local_repo_path, file_content_request)
                                if os.path.exists(file_full_path) and os.path.isfile(file_full_path):
                                    with open(file_full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        file_content_from_local = f.read()
                                    logger.info(f"Successfully read file from local repository: {file_content_request}")
                                else:
                                    logger.warning(f"File not found in local repository: {file_full_path}")
                            else:
                                logger.warning(f"Local repository path not available: {local_repo_path}")
                        else:
                            logger.warning("Database manager or repo paths not available")
                    except Exception as e:
                        logger.error(f"Error reading file from local repository: {str(e)}")
                    
                    # If file not found locally, try to get it from remote repository
                    if not file_content_from_local:
                        try:
                            file_content_from_local = get_file_content(
                                request.repo_url, 
                                file_content_request, 
                                request.type, 
                                request.token
                            )
                            logger.info(f"Successfully retrieved file from remote repository: {file_content_request}")
                        except Exception as e:
                            logger.warning(f"Could not retrieve file from remote repository: {str(e)}")
                
                # If it's a file listing query, get all unique file paths from the database
                elif is_file_listing_query:
                    logger.info("Detected file listing query - collecting all file paths")
                    all_file_paths = set()
                    for doc in request_rag.transformed_docs:
                        file_path = doc.meta_data.get('file_path', '')
                        if file_path and file_path != 'unknown' and file_path:
                            all_file_paths.add(file_path)
                    
                    # Sort files for better readability
                    sorted_files = sorted(all_file_paths)
                    
                    # Format the file list
                    file_list_text = f"## Complete File List\n\n**Total Files: {len(sorted_files)}**\n\n"
                    file_list_text += "### All Files:\n\n"
                    
                    # Group files by directory
                    files_by_dir = {}
                    for file_path in sorted_files:
                        dir_path = os.path.dirname(file_path) if os.path.dirname(file_path) else "."
                        if dir_path not in files_by_dir:
                            files_by_dir[dir_path] = []
                        files_by_dir[dir_path].append(os.path.basename(file_path))
                    
                    # Format by directory
                    for dir_path in sorted(files_by_dir.keys()):
                        if dir_path == ".":
                            file_list_text += "### Root Directory:\n\n"
                        else:
                            file_list_text += f"### {dir_path}/\n\n"
                        
                        for filename in sorted(files_by_dir[dir_path]):
                            full_path = filename if dir_path == "." else f"{dir_path}/{filename}"
                            file_list_text += f"- `{full_path}`\n"
                        file_list_text += "\n"
                    
                    # Also provide a flat list
                    file_list_text += "### Complete Path List:\n\n"
                    for file_path in sorted_files:
                        file_list_text += f"- `{file_path}`\n"
                    
                    context_text = file_list_text
                    logger.info(f"Collected {len(sorted_files)} unique file paths")
                
                # If file content was successfully read, skip RAG and use file content directly
                elif file_content_from_local:
                    logger.info(f"Using file content from repository for: {file_content_request}")
                    # File content will be added to prompt separately, no need for RAG context
                    context_text = ""
                
                # If filePath exists, modify the query for RAG to focus on the file
                elif request.filePath:
                    rag_query = f"Contexts related to {request.filePath}"
                    logger.info(f"Modified RAG query to focus on file: {request.filePath}")

                    # Try to perform RAG retrieval
                    try:
                        # This will use the actual RAG implementation
                        retrieved_documents = request_rag(rag_query, language=request.language)

                        if retrieved_documents and retrieved_documents[0].documents:
                            # Format context for the prompt in a more structured way
                            documents = retrieved_documents[0].documents
                            logger.info(f"Retrieved {len(documents)} documents")

                            # Group documents by file path
                            docs_by_file = {}
                            for doc in documents:
                                file_path = doc.meta_data.get('file_path', 'unknown')
                                if file_path not in docs_by_file:
                                    docs_by_file[file_path] = []
                                docs_by_file[file_path].append(doc)

                            # Format context text with file path grouping
                            context_parts = []
                            for file_path, docs in docs_by_file.items():
                                # Add file header with metadata
                                header = f"## File Path: {file_path}\n\n"
                                # Add document content
                                content = "\n\n".join([doc.text for doc in docs])

                                context_parts.append(f"{header}{content}")

                            # Join all parts with clear separation
                            context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                        else:
                            logger.warning("No documents retrieved from RAG")
                    except Exception as e:
                        logger.error(f"Error in RAG retrieval: {str(e)}")
                        # Continue without RAG if there's an error
                
                # Regular RAG query
                else:
                    rag_query = query
                    # Try to perform RAG retrieval
                    try:
                        # This will use the actual RAG implementation
                        retrieved_documents = request_rag(rag_query, language=request.language)

                        if retrieved_documents and retrieved_documents[0].documents:
                            # Format context for the prompt in a more structured way
                            documents = retrieved_documents[0].documents
                            logger.info(f"Retrieved {len(documents)} documents")

                            # Group documents by file path
                            docs_by_file = {}
                            for doc in documents:
                                file_path = doc.meta_data.get('file_path', 'unknown')
                                if file_path not in docs_by_file:
                                    docs_by_file[file_path] = []
                                docs_by_file[file_path].append(doc)

                            # Format context text with file path grouping
                            context_parts = []
                            for file_path, docs in docs_by_file.items():
                                # Add file header with metadata
                                header = f"## File Path: {file_path}\n\n"
                                # Add document content
                                content = "\n\n".join([doc.text for doc in docs])

                                context_parts.append(f"{header}{content}")

                            # Join all parts with clear separation
                            context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                        else:
                            logger.warning("No documents retrieved from RAG")
                    except Exception as e:
                        logger.error(f"Error in RAG retrieval: {str(e)}")
                        # Continue without RAG if there's an error

            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                context_text = ""

        # Get repository information
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

        # Determine repository type
        repo_type = request.type

        # Get language information
        language_code = request.language or configs["lang_config"]["default"]
        supported_langs = configs["lang_config"]["supported_languages"]
        language_name = supported_langs.get(language_code, "English")

        # Create system prompt
        if is_deep_research:
            # Check if this is the first iteration
            is_first_iteration = research_iteration == 1

            # Check if this is the final iteration
            is_final_iteration = research_iteration >= 5

            if is_first_iteration:
                system_prompt = DEEP_RESEARCH_FIRST_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    language_name=language_name
                )
            elif is_final_iteration:
                system_prompt = DEEP_RESEARCH_FINAL_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    research_iteration=research_iteration,
                    language_name=language_name
                )
            else:
                system_prompt = DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    research_iteration=research_iteration,
                    language_name=language_name
                )
        else:
            system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
                repo_type=repo_type,
                repo_url=repo_url,
                repo_name=repo_name,
                language_name=language_name
            )

        # Fetch file content if provided
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"Successfully retrieved content for file: {request.filePath}")
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")
                # Continue without file content if there's an error

        # Format conversation history
        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
                conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"

        # Create the prompt with context
        prompt = f"/no_think {system_prompt}\n\n"

        if conversation_history:
            prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"
        
        # Add file context if we have a file from current query or conversation history
        current_file_context = file_content_request or context_file_path
        if current_file_context:
            prompt += f"<file_context>IMPORTANT: The current conversation is about the file: {current_file_context}. When the user asks questions like '都是哪些', '有哪些', '里面有多少个', 'what are they', etc., they are referring to THIS file ({current_file_context}), NOT other files mentioned in the conversation history. Always answer questions in the context of {current_file_context}.</file_context>\n\n"

        # Check if filePath is provided and fetch file content if it exists
        if file_content:
            # Add file content to the prompt after conversation history
            prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

        # If we detected a file content request and read it, add it to prompt
        if file_content_from_local:
            prompt += f"<requestedFileContent path=\"{file_content_request}\">\n{file_content_from_local}\n</requestedFileContent>\n\n"
            logger.info(f"Added requested file content to prompt: {file_content_request}")
            # Add instruction to show the code
            prompt += f"<instruction>The user is requesting the code for the file: {file_content_request}. Please provide the complete code content of this file.</instruction>\n\n"

        # Only include context if it's not empty
        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"
        if context_text.strip():
            prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
        else:
            # Add a note that we're skipping RAG due to size constraints or because it's the isolated API
            logger.info("No context available from RAG")
            prompt += "<note>Answering without retrieval augmentation.</note>\n\n"

        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        model_config = get_model_config(request.provider, request.model)["model_kwargs"]

        if request.provider == "openai":
            logger.info(f"Using Openai protocol with model: {request.model}")

            # Check if an API key is set for Openai
            if not OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not configured, but continuing with request")
                # We'll let the OpenAIClient handle this and return an error message

            # Initialize Openai client
            model = OpenAIClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"]
            }
            # Only add top_p if it exists in the model config
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )


        # Create a streaming response
        async def response_stream():
            try:
                if request.provider == "openai":
                    try:
                        # Get the response and handle it properly using the previously created api_kwargs
                        logger.info("Making Openai API call")
                        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                        # Handle streaming response from Openai
                        async for chunk in response:
                           choices = getattr(chunk, "choices", [])
                           if len(choices) > 0:
                               delta = getattr(choices[0], "delta", None)
                               if delta is not None:
                                    text = getattr(delta, "content", None)
                                    if text is not None:
                                        yield text
                    except Exception as e_openai:
                        logger.error(f"Error with Openai API: {str(e_openai)}")
                        yield f"\nError with Openai API: {str(e_openai)}\n\nPlease check that you have set the OPENAI_API_KEY environment variable with a valid API key."


            except Exception as e_outer:
                logger.error(f"Error in streaming response: {str(e_outer)}")
                error_message = str(e_outer)

                # Check for token limit errors
                if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                    # If we hit a token limit error, try again without context
                    logger.warning("Token limit exceeded, retrying without context")
                    try:
                        # Create a simplified prompt without context
                        simplified_prompt = f"/no_think {system_prompt}\n\n"
                        if conversation_history:
                            simplified_prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"
                        
                        # Add file context if we have a file from current query or conversation history
                        current_file_context = file_content_request or context_file_path
                        if current_file_context:
                            simplified_prompt += f"<file_context>IMPORTANT: The current conversation is about the file: {current_file_context}. When the user asks questions like '都是哪些', '有哪些', '里面有多少个', 'what are they', etc., they are referring to THIS file ({current_file_context}), NOT other files mentioned in the conversation history. Always answer questions in the context of {current_file_context}.</file_context>\n\n"

                        # Include file content in the fallback prompt if it was retrieved
                        if request.filePath and file_content:
                            simplified_prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"
                        
                        # Include requested file content if available
                        if file_content_from_local:
                            simplified_prompt += f"<requestedFileContent path=\"{file_content_request}\">\n{file_content_from_local}\n</requestedFileContent>\n\n"
                            simplified_prompt += f"<instruction>The user is requesting the code for the file: {file_content_request}. Please provide the complete code content of this file.</instruction>\n\n"

                        simplified_prompt += "<note>Answering without retrieval augmentation due to input size constraints.</note>\n\n"
                        simplified_prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

                        if request.provider == "openai":
                            try:
                                # Create new api_kwargs with the simplified prompt
                                fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                    input=simplified_prompt,
                                    model_kwargs=model_kwargs,
                                    model_type=ModelType.LLM
                                )

                                # Get the response using the simplified prompt
                                logger.info("Making fallback Openai API call")
                                fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                                # Handle streaming fallback_response from Openai
                                async for chunk in fallback_response:
                                    text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                                    yield text
                            except Exception as e_fallback:
                                logger.error(f"Error with Openai API fallback: {str(e_fallback)}")
                                yield f"\nError with Openai API fallback: {str(e_fallback)}\n\nPlease check that you have set the OPENAI_API_KEY environment variable with a valid API key."

                    except Exception as e2:
                        logger.error(f"Error in fallback streaming response: {str(e2)}")
                        yield f"\nI apologize, but your request is too large for me to process. Please try a shorter query or break it into smaller parts."
                else:
                    # For other errors, return the error message
                    yield f"\nError: {error_message}"

        # Return streaming response
        return StreamingResponse(response_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e_handler:
        error_msg = f"Error in streaming chat completion: {str(e_handler)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"status": "API is running", "message": "Navigate to /docs for API documentation"}