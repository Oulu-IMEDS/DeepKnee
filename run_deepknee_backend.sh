mkdir -p logs
KNEEL_ADDR=http://kneel:5000 python -m ouludeepknee.inference.app \
                   --snapshots_path snapshots_knee_grading/ \
                   --device cpu --deploy True \
                   --port 5001 \
                   --logs logs/deepknee-cpu.log