# MLflow
## for dags
MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/chest-Disease-Classification-MLflow-DVC.mlflow
MLFLOW_TRACKING_USERNAME=entbappy
MLFLOW_TRACKING_PASSWORD=6824692c47a4545eac5b10041d5c8edbcef0

# AWS Setup

sudo apt update
sudo apt install python3-pip

sudo apt install pipenv
sudo apt install python3-virtualenv

mkdir mlflow
cd mlflow

pipenv install mlflow
pipenv install awscli
pipenv install boto3

pipenv shell

# Then set aws credentials
aws configure

#Finally
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-bucket1

#open Public IPv4 DNS to the port 5000

#set uri in your local terminal and in your code
export MLFLOW_TRACKING_URI=(http://ec2-3-106-53-6.ap-southeast-2.compute.amazonaws.com:5000/)

export MLFLOW_TRACKING_URI=http://ec2-3-106-53-6.ap-southeast-2.compute.amazonaws.com:5000/
