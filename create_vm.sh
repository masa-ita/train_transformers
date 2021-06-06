gcloud compute --project=${PROJECT_ID} instances create train-transformers \
    --zone=us-central1-a  \
    --machine-type=n1-standard-16  \
    --image-family=torch-xla \
    --image-project=ml-images  \
    --boot-disk-size=200GB \
    --scopes=https://www.googleapis.com/auth/cloud-platform

