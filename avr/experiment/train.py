import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch import hub

from avr.config import register_omega_conf_resolvers
from avr.wandb import WandbClient


@hydra.main(config_path="../../config", config_name="default", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision(cfg.torch.float32_matmul_precision)

    hub.set_dir(cfg.model_dir)
    pl.seed_everything(cfg.seed)
    module = instantiate(cfg["avr"]["task"][cfg.avr.problem][cfg.avr.dataset], cfg)

    if "wandb_checkpoint" in cfg:
        client = WandbClient()
        checkpoint_path = client.download_checkpoint_by_run_name(cfg.wandb_checkpoint)
        module = module.load_from_checkpoint(
            checkpoint_path,
            cfg=cfg,
            strict=False,
            **{
                k: v
                for k, v in cfg["avr"]["task"][cfg.avr.problem][cfg.avr.dataset].items()
                if k != "_target_"
            }
        )

    data_module = instantiate(cfg.avr.datamodule, cfg)
    trainer: pl.Trainer = instantiate(cfg.pytorch_lightning.trainer)
    trainer.fit(module, data_module)
    trainer.test(module, data_module)


if __name__ == "__main__":
    register_omega_conf_resolvers()
    load_dotenv()
    main()
