from time import localtime, strftime
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://0.0.0.0:8265")

kick_off_pytorch_benchmark = (
    # Clone. If ray is already present, don't clone again.
    "git clone https://github.com/wxharry/cml_proj2 || true;"
    # Run the training code.
    "sh cml_proj2/train/run_job.sh;"
)


submission_id = client.submit_job(
    entrypoint=kick_off_pytorch_benchmark,
    job_id=strftime('train-%H-%M-%S')
)

print("Use the following command to follow this Job's logs:")
print(f"ray job logs '{submission_id}' --address http://0.0.0.0:8265 --follow")
