# Use AWS Lambda's Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11


# Install required Python libraries
RUN pip install --no-cache-dir pysqlite3-binary supabase boto3 langchain-openai langchain-community openai chromadb

# Copy your Lambda function code
COPY jihyun_rag_endpoint.py ${LAMBDA_TASK_ROOT}

# Set the command to run your Lambda function
CMD ["jihyun_rag_endpoint.lambda_handler"]
