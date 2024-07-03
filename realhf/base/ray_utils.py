import subprocess
import os


def check_ray_availability():
    return (
        int(
            subprocess.run(
                ["ray", "--help"],
                stdout=open(os.devnull, "wb"),
                stderr=open(os.devnull, "wb"),
            ).returncode
        )
        == 0
    )
