# Experiment log

Paper: One self-configurable model to solve many Abstract Visual Reasoning problems	

## Single-task learning (STL)

### G-set
```bash
for model in 'wren' 'lstm' 'scl_sal' 'scar' 'copinet' 'sran' 'relbase' 'scl'; do
  ./scripts/train.sh +experiment=stl_gset avr/model=${model} avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
done
```

### PGM
```bash
for model in 'wren' 'lstm' 'scl_sal' 'scar' 'copinet' 'sran' 'relbase' 'scl'; do
  ./scripts/train.sh +experiment=stl_pgm avr/model=${model} avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
done
```

### I-RAVEN
```bash
for model in 'wren' 'lstm' 'scl_sal' 'scar' 'copinet' 'sran' 'relbase' 'scl'; do
  ./scripts/train.sh +experiment=stl_raven avr/model=${model} avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
done
```

### VAP
```bash
for model in 'wren' 'lstm' 'scl_sal' 'scar'; do
  ./scripts/train.sh +experiment=stl_vap avr/model=${model} avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
done
```

### O3
```bash
for model in 'wren_o3' 'lstm_o3' 'scl_sal_o3' 'scar_o3'; do
  ./scripts/train.sh +experiment=stl_o3 avr/model=${model} avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
done
```

## Multi-task learning (MTL)

### Pre-training
```bash
for model in 'wren' 'lstm' 'scl_sal' 'scar' 'copinet' 'sran' 'relbase' 'scl'; do
  ./scripts/train.sh +experiment=mtl_rpm avr/model=${model} avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE} avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE} avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
done

for model in 'wren' 'lstm' 'scl_sal' 'scar'; do
  ./scripts/train.sh +experiment=mtl_rpmvap avr/model=${model} avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE} avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE} avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE} avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
done
```

### Fine-tuning
```bash
# WReN
./scripts/train.sh +experiment=stl_gset avr/model=wren +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=wren +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=wren +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_vap avr/model=wren +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

./scripts/train.sh +experiment=stl_gset avr/model=wren +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=wren +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=wren +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_vap avr/model=wren +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

# RNN
./scripts/train.sh +experiment=stl_gset avr/model=lstm +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=lstm +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=lstm +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_vap avr/model=lstm +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

./scripts/train.sh +experiment=stl_gset avr/model=lstm +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=lstm +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=lstm +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_vap avr/model=lstm +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

# SCL+SAL
./scripts/train.sh +experiment=stl_gset avr/model=scl_sal +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=scl_sal +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=scl_sal +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_vap avr/model=scl_sal +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

./scripts/train.sh +experiment=stl_gset avr/model=scl_sal +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=scl_sal +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=scl_sal +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_vap avr/model=scl_sal +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

# SCAR
./scripts/train.sh +experiment=stl_gset avr/model=scar +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=scar +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=scar +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_vap avr/model=scar +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

./scripts/train.sh +experiment=stl_gset avr/model=scar +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=scar +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=scar +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_vap avr/model=scar +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

# CoPINet
./scripts/train.sh +experiment=stl_gset avr/model=copinet +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=copinet +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=copinet +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

# SRAN
./scripts/train.sh +experiment=stl_gset avr/model=sran +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=sran +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=sran +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

# RelBase
./scripts/train.sh +experiment=stl_gset avr/model=relbase +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=relbase +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=relbase +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}

# SCL
./scripts/train.sh +experiment=stl_gset avr/model=scl +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_pgm avr/model=scl +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_raven avr/model=scl +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
```

## Transfer learning

### RPMs -> 03
```bash
./scripts/train.sh +experiment=stl_o3 avr/model=wren_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_o3 avr/model=lstm_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_o3 avr/model=scl_sal_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_o3 avr/model=scar_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
```

### VAPs -> 03
```bash
./scripts/train.sh +experiment=stl_o3 avr/model=wren_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_o3 avr/model=lstm_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_o3 avr/model=scl_sal_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_o3 avr/model=scar_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
```

### RPMs + VAPs -> 03
```bash
./scripts/train.sh +experiment=stl_o3 avr/model=wren_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_o3 avr/model=lstm_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_o3 avr/model=scl_sal_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
./scripts/train.sh +experiment=stl_o3 avr/model=scar_o3 +wandb_checkpoint=<replace-with-checkpoint-name> avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
```

## Ablation study

```bash
for model in 'scar_rn' 'scar_lstm'; do
  ./scripts/train.sh +experiment=stl_gset avr/model=${model} avr.data.rpm.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
  ./scripts/train.sh +experiment=stl_pgm avr/model=${model} avr.data.rpm.pgm.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
  ./scripts/train.sh +experiment=stl_raven avr/model=${model} avr.data.rpm.raven.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
  ./scripts/train.sh +experiment=stl_vap avr/model=${model} avr.data.vap.hill2019learning.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
done

for model in 'scar_rn_o3' 'scar_lstm_o3'; do
  ./scripts/train.sh +experiment=stl_o3 avr/model=${model} avr.data.ooo.deepiq.train.augmentor_factory=\${AugmentorFactory:SIMPLE}
done
```
