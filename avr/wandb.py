import os
from typing import Optional

import wandb
from wandb.apis.public import Run


class WandbClient:
    def __init__(self, project_name: str = "avr"):
        self.api = wandb.Api()
        self.project_name = project_name

    def get_run_by_name(self, run_name: str) -> Run:
        runs = self.api.runs(self.project_name, filters={"display_name": run_name})
        if len(runs) == 1:
            return runs[0]
        else:
            raise ValueError(
                f"Query for runs with display_name='{run_name}' returned {len(runs)} results."
            )

    def download_checkpoint_by_artifact_name(self, artifact_name: str) -> str:
        artifact = self.api.artifact(
            f"{self.project_name}/{artifact_name}", type="model"
        )
        return self.download_ckeckpoint(artifact)

    def download_checkpoint_by_run_name(
        self, run_name: str, artifact_version: Optional[str] = None
    ) -> str:
        run = self.get_run_by_name(run_name)
        artifacts = run.logged_artifacts()
        model_artifacts = [a for a in artifacts if a.type == "model"]
        if model_artifacts:
            if artifact_version:
                artifact = None
                for a in model_artifacts:
                    if a.name.endswith(artifact_version):
                        print(
                            f"Run {run.name} has {len(model_artifacts)} model artifacts. Using provided version: {artifact_version}."
                        )
                        artifact = a
                        break
                if not artifact:
                    raise ValueError(
                        f"Run {run.name} has no artifact with version {artifact_version}"
                    )
            else:
                print(
                    f"Run {run.name} has {len(model_artifacts)} model artifacts. Using the last one."
                )
                artifact = model_artifacts[-1]
            return self.download_ckeckpoint(artifact)
        else:
            raise ValueError(f"Run {run.name} has no artifacts with type model.")

    @staticmethod
    def download_ckeckpoint(artifact: wandb.Artifact) -> str:
        checkpoint_dir = artifact.download()
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
        print(
            f"Success: downloaded wandb artifact={artifact.name} to path={checkpoint_path}"
        )
        return checkpoint_path
