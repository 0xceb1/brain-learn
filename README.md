# brain-learn

A genetic programming framework for the [WorldQuant BRAIN platform](https://platform.worldquantbrain.com/), inspired by [gplearn](https://github.com/trevorstephens/gplearn).

## Usage

1. Rename `.env.example` to `.env` and modify the `USERNAME` and `PASSWORD` to your own.
2. Modify `main.py` to set your desired parameters for the genetic programming run (e.g., population size, generations, simulation settings).
3. (Optional) You can provide an initial population of expressions (signals) to kickstart the evolution process. Refer to the implementation in `main.py` or `src/genetic.py` for how to load or specify this initial population.

## Customization

You can customize the building blocks of the genetic programming process:

*   **Operators and Terminals:** Add or modify operators (functions like `add`, `ts_rank`) and terminals (input features like `close`, `volume`) in `src/function.py`.
*   **Weights:** Adjust the `weight` parameter for `Operator` and `Terminal` instances in `src/function.py` to influence their probability of being selected during the evolutionary process. Higher weights mean higher probability.

## Disclaimer

Notice: This codebase is experimental and intended solely for personal use. It is provided 'AS IS', without representation or warranty of any kind. Liability for any use or reliance upon this software is expressly disclaimed. USE AT YOUR OWN RISK.




