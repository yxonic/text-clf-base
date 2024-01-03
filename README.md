# text-clf-base

**Table of Contents**

- [Installation](#installation)
- [Development](#development)
- [License](#license)

## Installation

**Change project name to your own before using this template**:

```console
mv src/text_clf_base src/YOUR_PROJECT_NAME
fd -t f . -x sed -i '' 's/text_clf_base/YOUR_PROJECT_NAME/g'
```

Then, to run this project, install the package in editable mode:

```console
pip install -e .
```

## Development

To set up a development environment for the package, first install `hatch`:

```console
pipx install hatch  # or: brew install hatch
```

Then, run the following command in your terminal:

```console
hatch shell
```

## License

`text-clf-base` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
