gcloud compute tpus create train-transformers \
    --zone=us-central1-a \
    --network=default \
    --version=pytorch-1.8 \
    --accelerator-type=v3-8

gcloud compute tpus list --zone=us-central1-a
