import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# Read the Python script content
with open("rl_stock_trading.py", "r") as f:
    script_content = f.read()

# Create a new notebook object with metadata
nb = new_notebook(
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0rc1"
        }
    }
)

# Add markdown introduction
nb.cells.append(new_markdown_cell("# Reinforcement Learning for Stock Trading in Data-Scarce Environments\n\nThis notebook implements a reinforcement learning (RL) solution for stock trading.\n"))

# Add the code content as a single code cell
nb.cells.append(new_code_cell(script_content))

# Write the notebook to a file
with open("reinforcement_learning_stock_trading_final.ipynb", "w") as f:
    nbformat.write(nb, f)

print("Notebook \'reinforcement_learning_stock_trading_final.ipynb\' created successfully.")


