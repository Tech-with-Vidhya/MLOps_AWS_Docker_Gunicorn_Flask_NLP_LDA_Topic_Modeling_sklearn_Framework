# MLOps_AWS_Docker_Gunicorn_Flask_NLP_LDA_Topic_Modeling_sklearn_Framework

<h3><b><u>Introduction:</u></b></h3>

This project covers the end to end automated ML+MLOps pipeline implementation of a NLP Text Analytics Solution for Topic Modeling using Latent Dirichlet allocation (LDA) Algorithm, deployed into Amazon Web Srvices (AWS).

The main objective of the NLP Model is to extract or identify a dominant topic from each document and perform topic modeling using LDA technique.

<h3><b><u>Dataset:</u></b></h3>

The dataset has odd 25000 documents where words are of various nature such as Noun,Adjective,Verb,Preposition and many more. Even the length of documents varies vastly from having a minimum number of words in the range around 40 to maximum number of words in the range around 500. 

Complete data is split 90% in the training and the rest 10% to get an idea how to predict a topic on unseen documents.

<h3><b><u>Project Implementation Steps using AWS:</u></b></h3>

1. Data Exploration and Analysis
2. Data Pre-processing
3. Model Training
4. Model Evaluation and Validation
5. Model Performance Metrics Measures
6. Packaging the Finalised Model as a Pythin Flask Application
7. Dockerized the Application using DockerFile
8. Imported the Docker Image into AWS ECR
9. Configured the AWS Code Pipeline with Code Commit, Code Build and Code Deploy
10. Configured the AWS EC2 Instance
11. Enabled Load Balancing Capability on the AWS EC2 Instance
12. Deployed the Application into AWS EC2 Instance
13. Verified the Deployed Application Endpoint for Model Inferences

<h3><b><u>Tools & Technologies:</u></b></h3>
Python, Flask, Gunicorn, AWS S3, AWS ECR, AWS Code Commit, AWS Code Build, AWS Code Deploy, AWS Code Pipeline, boto3, AWS CLI, AWS EC2, AWS Load Balancing, AWS IAM.


