# src/rag/api/main.py
"""FastAPI application."""
from contextlib import asynccontextmanager  # Used to manage startup and shutdown events
from fastapi import FastAPI, Request  # FastAPI framework and Request object for handling HTTP requests
from fastapi.middleware.cors import CORSMiddleware  # Middleware to handle Cross-Origin Resource Sharing (allows frontend apps to call this API)
from fastapi.responses import JSONResponse  # Used to return JSON formatted responses
import time  # Used to measure request processing time

from rag.api.routes import documents, query, health  # Import route modules that define API endpoints
from rag.exceptions import RAGException  # Custom exception class for RAG-specific errors
from rag.logging_config import setup_logging, get_logger  # Functions to configure and get loggers
from rag.config import settings  # Application configuration settings


@asynccontextmanager  # Decorator that allows this function to manage startup and shutdown logic
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup - code before 'yield' runs when the application starts
    setup_logging(json_format=True)  # Configure logging to output in JSON format for better log parsing
    logger = get_logger("api")  # Get a logger instance for the API module
    logger.info("application_starting", version="0.2.0")  # Log that the application is starting with version info

    yield  # This separates startup code (above) from shutdown code (below)

    # Shutdown - code after 'yield' runs when the application is stopping
    logger.info("application_stopping")  # Log that the application is shutting down


app = FastAPI(  # Create the main FastAPI application instance
    title="RAG System API",  # API title shown in documentation
    description="REST API for Retrieval-Augmented Generation",  # API description shown in documentation
    version="0.2.0",  # API version number
    lifespan=lifespan,  # Attach the lifespan function to handle startup/shutdown events
)

# CORS middleware - allows web browsers to make requests from different domains
app.add_middleware(  # Add middleware to the application (middleware processes requests before they reach endpoints)
    CORSMiddleware,  # Middleware that handles Cross-Origin Resource Sharing
    allow_origins=["*"],  # Allow requests from any origin ("*" means all domains) - Configure for production to restrict specific domains
    allow_credentials=True,  # Allow cookies and authentication headers in cross-origin requests
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all HTTP headers in requests
)


# Request timing middleware - measures how long each request takes to process
@app.middleware("http")  # Decorator to register this function as HTTP middleware
async def add_timing_header(request: Request, call_next):  # call_next is a function that processes the request
    start_time = time.time()  # Record the current time before processing the request
    response = await call_next(request)  # Process the request and get the response (await pauses until processing is done)
    process_time = time.time() - start_time  # Calculate how long the request took by subtracting start time from current time
    response.headers["X-Process-Time"] = f"{process_time:.3f}"  # Add a custom header to the response showing processing time in seconds (3 decimal places)
    return response  # Return the response with the added timing header


# Exception handlers - functions that handle errors that occur during request processing
@app.exception_handler(RAGException)  # This handler catches RAGException errors (custom errors from our RAG system)
async def rag_exception_handler(request: Request, exc: RAGException):  # exc is the exception object that was raised
    logger = get_logger("api")  # Get the logger to record the error
    logger.error(  # Log the error with details
        "rag_exception",  # Error type label
        error=exc.__class__.__name__,  # The name of the exception class (e.g., "RAGException")
        message=exc.message,  # The error message from the exception
        details=exc.details,  # Additional details about the error
        path=str(request.url),  # The URL path where the error occurred
    )
    return JSONResponse(  # Return a JSON response to the client
        status_code=400,  # HTTP status code 400 means "Bad Request" (client error)
        content={  # The JSON content to send back
            "error": exc.__class__.__name__,  # Error type name
            "message": exc.message,  # Human-readable error message
            "details": exc.details,  # Additional error information
        },
    )


@app.exception_handler(Exception)  # This handler catches all other exceptions that aren't RAGException
async def general_exception_handler(request: Request, exc: Exception):  # exc is any Python exception
    logger = get_logger("api")  # Get the logger to record the error
    logger.exception(  # Log the exception with full traceback (stack trace)
        "unhandled_exception",  # Error type label for unexpected errors
        error=str(exc),  # Convert the exception to a string for logging
        path=str(request.url),  # The URL path where the error occurred
    )
    return JSONResponse(  # Return a JSON response to the client
        status_code=500,  # HTTP status code 500 means "Internal Server Error" (server-side error)
        content={  # The JSON content to send back
            "error": "InternalServerError",  # Generic error type name
            "message": "An unexpected error occurred",  # Generic error message (don't expose internal details to users)
            "details": {},  # Empty details object (don't expose sensitive error information)
        },
    )


# Include routers - attach route modules to the main app (routers group related endpoints together)
app.include_router(health.router, prefix="/api/v1", tags=["Health"])  # Add health check endpoints (e.g., /api/v1/health) with "Health" tag in docs
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])  # Add document management endpoints (e.g., /api/v1/documents) with "Documents" tag in docs
app.include_router(query.router, prefix="/api/v1", tags=["Query"])  # Add query/search endpoints (e.g., /api/v1/query) with "Query" tag in docs


@app.get("/")  # Decorator that registers this function as a GET endpoint at the root path ("/")
async def root():  # Function that handles requests to the root URL
    """Root endpoint."""
    return {  # Return a dictionary that FastAPI automatically converts to JSON
        "name": "RAG System API",  # The name of the API
        "version": "0.2.0",  # The current version number
        "docs": "/docs",  # URL path where users can find the interactive API documentation (Swagger UI)
    }
    