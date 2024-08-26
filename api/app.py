import os
import json
import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')



def send_message(recipient_number, document_link, filename, caption="Dear Customer, Please find the attached document."):
    url = 'https://graph.facebook.com/v20.0/337483972792928/messages'
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_number,
        "type": "document",
        "document": {
            "link": document_link,
            "filename": filename,
            "caption": caption
        }
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # Log the response for debugging
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")

        return response.status_code, response.json()

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return 500, {'error': 'Failed to send message'}

def create_document_embedding(document_link,recipient_number):
    try:
        # Step 1: Download the PDF document
        pdf_response = requests.get(document_link)
        
        if pdf_response.status_code != 200:
            return 500, {'error': 'Failed to download the document for embeddings'}
        
        # Step 2: Save PDF content to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(pdf_response.content)
            temp_pdf_path = temp_pdf.name

        # Step 3: Load PDF using PyPDFLoader
        try:
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()    
        except Exception as e:
            return 500, {'error': f'Failed to load the PDF: {e}'}

        # Step 4: Split the documents into chunks
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.split_documents(documents)
        except Exception as e:
            return 500, {'error': f'Failed to split the document into chunks: {e}'}

        # Step 5: Prepare data for embedding
        data = [
            {"id": f"vec{i+1}", "text": doc.page_content}
            for i, doc in enumerate(documents)
        ]
        
        # Step 6: Initialize Pinecone and pinecoe index
        try:
            pc = PineconeGRPC(api_key=PINECONE_API_KEY)
            index = pc.Index('serverless-index-1')
        except Exception as e:
            return 500, {'error': f'Failed to initialize pinceone index: {e}'}

        # Step 7: Generate embeddings
        try:
            embeddings = pc.inference.embed(
                "multilingual-e5-large",
                inputs=[d['text'] for d in data],
                parameters={
                    "input_type": "passage"
                }
            )
        except Exception as e:
            return 500, {'error': f'Failed to generate embeddings: {e}'}
        
        # Step 8: Prepare the embeddings for Pinecone upload
        try:
            vectors = []
            for d, e in zip(data, embeddings):
                vectors.append({
                    "id": d['id'],
                    "values": e['values'],
                    "metadata": {'text': d['text']}
                })
        except Exception as e:
            return 500, {'error': f'Failed to prepare embeddings for upload: {e}'}
        
        # Step 9: Upload the embeddings to Pinecone
        try:
            index.upsert(
                vectors=vectors,
                namespace="ns1"
            )
        except Exception as e:
            return 500, {'error': f'Failed to upload embeddings to Pinecone: {e}'}

        return 200, {'message': 'Document embeddings created and uploaded successfully'}
    
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return 500, {'error': 'Internal Server Error'}
    
# ---------------

def lambda_handler(event, context):
    try: 
        #!ACCESS_TOKEN = os.getenv('ACCESS_TOKEN') //Commenting temporarily
        body = event.get('body')
    
        if not body:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Request body is missing'})
            }
        
        # Parse the JSON body
        body = json.loads(body)
        
        # Extract parameters from the body
        recipient_number = body.get('recipient_number')
        document_link = body.get('document_link')
        filename = body.get('filename')
        caption = body.get('caption', "Dear Customer, Please find the attached document.")

        # Check if recipient number is provided
        if not recipient_number:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'recipient_number is required'})
            }

        # Step 1: Create document embeddings
        embedding_status_code, embedding_response_body = create_document_embedding(document_link, recipient_number)
        print(embedding_status_code,embedding_response_body)
        # If embedding creation fails, return the error response
        if embedding_status_code != 200:
            return {
                'statusCode': embedding_status_code,
                'body': json.dumps(embedding_response_body)
            }
        
        # Step 2: Send the message (if embedding creation is successful)
        message_status_code, message_response_body = send_message(recipient_number, document_link, filename, caption)
        
        print("The message has been sent",message_status_code,message_response_body)
        # Return the response from the message sending
        return {
            'statusCode': message_status_code,
            'body': json.dumps(message_response_body)
        }

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON in request body'})
        }

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal Server Error'})
        }
