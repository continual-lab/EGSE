# Expand Only When Necessary: Expressibility-Guided Modular Continual Learning

Code for anonymous research submission.

## Overview

Evolutionary approach to continual learning with dynamic modular architecture growth.

**Key Features:**
- Dynamic module pool expansion
- CKA-based similarity for reuse decisions  
- PPO for composition search
- Task-specific routing probabilities

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.12+

## Quick Start

```python
from evo_trainer import EvoTrainer

config = {
    'device': 'cuda',
    'hidden_dim': 192,
    'num_classes': 100,
    'max_steps': 4,
    'module_epochs': 1000,
    'rl_epochs': 100,
    # ... see example_config.py
}

trainer = EvoTrainer(config)
for task_id in range(10):
    trainer.setup_data_loaders(task_id)
    trainer.train_task(task_id)
```

## Project Structure

```
├── controller/      # RL components
├── module/          # Expert modules
├── graph/           # Graph management
├── utils/           # Utilities
├── log/             # Logging
├── test.py          # Evaluation
└── evo_trainer.py   # Main trainer
```

## Method

1. **Module Discovery**: Train candidate → Compute CKA similarity → Reuse or create
2. **Composition Search**: PPO to find optimal module combinations
3. **Task Training**: Train with discovered architecture
4. **Evaluation**: Test on all previous tasks

## Key Parameters

| Param | Description | Default |
|-------|-------------|---------|
| hidden_dim | Feature dimension | 192 |
| max_steps | Module positions | 4 |
| module_epochs | Training epochs | 1000 |
| rl_epochs | RL epochs | 100 |
| repeat | Trajectories | 10 |

## Citation

## License

MIT
