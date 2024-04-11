srun --container-image=llm/llm-gpu \
    --nodelist=QH-com48 \
    --container-mounts=/lustre:/lustre -N1 \
    --pty -c99 --mem=900G --gpus=8 --container-mount-home bash
# --exclude=QH-com[01-14],QH-com29,QH-com35 \