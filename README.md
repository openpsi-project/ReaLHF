# A Distributed System for LLM RLHF

## Contributing

Before reading the code, keep in mind that:
1. Everything in [api/config.py](api/config.py) are configurations. They are used to configurate your experiment.
2. Other classes in [api directory](api/) are abstract methods. They represent necessary components for the system to run.
3. Classes in [api/config.py](api/config.py) and other scripts in [api directory](api/) may have same class names. 

To run the code, see `docs/user_guide/00_quickstart.md`.

To understand difference between `model`, `model backend` and `model interface`, read [this doc](docs/user_guide/02_model.md).

If you want to contribute to the codebase (e.g., new datasets/models/algorithms), please open an MR and @fuwei for code review.
