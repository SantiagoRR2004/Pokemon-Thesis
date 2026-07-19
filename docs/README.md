# Pokémon

This is the repository that contains the code and documentation for my thesis, titled "**Reinforcement Learning in Strategy Games: Agent for Pokémon Showdown**". The objective was to investigate how to create an agent for playing the *gen9randombattle* format in Pokémon Showdown using Reinforcement Learning. It was publicly defended before the examination committee on 1 July 2026.

Only the LaTeX source files are included in this repository. To compile them into the thesis documentation (`README-en.pdf` and `README-es.pdf`), you need to have a LaTeX distribution installed on your computer. I recommend installing the full TeX Live distribution to not have any headaches with missing packages. You can install it with the following command:

```bash
sudo apt install texlive-full
```

Once you have the LaTeX distribution installed, clone (or download) this repository this repository and save the main LaTeX file: [`README.tex`](./README.tex), or any of the files in [`sections/`](./sections/) in VSCode. The project is configured to compile automatically on save using the VS Code settings included in the repository. These settings should also download all the required extensions, such as `James-Yu.latex-workshop`, when the folder is opened in VSCode.
