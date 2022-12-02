# datar-pandas

The pandas backend for [datar][1].

## Installation

```bash
pip install -U datar-pandas
# or
pip install -U datar[pandas]
```

## Usage

```python
from datar import f
# Without the backend: NotImplementedByCurrentBackendError
from datar.data import iris
from datar.dplyr import mutate

# Without the backend: NotImplementedByCurrentBackendError
iris >> mutate(sepal_ratio = f.Sepal_Width / f.Sepal_Length)
```

[1]: https://github.com/pwwang/datar
