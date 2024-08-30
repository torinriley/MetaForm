# Contributing to MetaForm

We are excited that you are interested in contributing to MetaForm! Your contributions help make MetaForm a better and more powerful tool for everyone. This guide will help you get started with contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Contributing Code](#contributing-code)
  - [Improving Documentation](#improving-documentation)
- [Development Process](#development-process)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
  - [Workflow](#workflow)
- [Style Guide](#style-guide)
- [License](#license)



## How Can I Contribute?

### Reporting Bugs

If you find a bug in the MetaForm library, please report it by creating an issue in our [GitHub issue tracker](https://github.com/torinriley/MetaForm/issues). When reporting a bug, please include as much detail as possible to help us understand and reproduce the issue.

### Suggesting Enhancements

We welcome suggestions for new features or improvements to existing features. To suggest an enhancement, please open a [GitHub issue](https://github.com/torinriley/MetaForm/issues) and describe your idea in detail. Be sure to explain why the feature would be beneficial to the project.

### Contributing Code

We welcome contributions in the form of code! If you would like to contribute code to MetaForm, please follow these steps:

1. **Fork the repository**: Click the "Fork" button at the top right corner of the [repository page](https://github.com/torinriley/metaform).
2. **Create a new branch**: Create a branch for your feature or bugfix (`git checkout -b feature-branch`).
3. **Make your changes**: Implement your feature or bugfix.
4. **Write tests**: Ensure your code is well-tested.
5. **Commit your changes**: Use descriptive commit messages (`git commit -m 'Add new feature'`).
6. **Push to your branch**: Push the changes to your forked repository (`git push origin feature-branch`).
7. **Create a Pull Request**: Open a pull request to the `main` branch of the original repository.

### Improving Documentation

Documentation is key to the success of any open-source project. If you find areas in the documentation that can be improved or expanded, feel free to contribute. You can update the documentation directly and follow the same process as contributing code.

## Development Process

### Setting Up the Development Environment

1. **Clone the repository**:
    ```bash
    git clone https://github.com/torinriley/metaform.git
    cd metaform
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run tests**:
    ```bash
    pytest
    ```
    Ensure that all tests pass before making your changes.
     ### NOTE - Tests are still under development, comding soon ###

### Workflow

1. **Work on your branch**: Develop your feature or fix in a separate branch.
2. **Keep your branch up to date**: Periodically rebase your branch against the `main` branch to stay up to date.
3. **Write meaningful commit messages**: Commit messages should be clear and descriptive.
4. **Submit a pull request**: Once you are satisfied with your changes, open a pull request. 

## Style Guide

- **Code Style**: Follow the PEP 8 style guide for Python code.
- **Docstrings**: Use Google-style docstrings for documenting your functions and classes.
- **Testing**: Ensure that your code is covered by tests. Use `pytest` for testing.

## License

By contributing to MetaForm, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to MetaForm!
