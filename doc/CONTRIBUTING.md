# Contributing

## Code quality checks

### pre-commit

[pre-commit](https://pre-commit.com/) is a tool for automating code quality
checks at commit-time, preventing problems from ever being introduced to the
repository. The same config can be used to run these checks in
[pre-commit.ci](https://pre-commit.ci/), so even contributors who don't install
`pre-commit` will benefit.

To setup `pre-commit`, run:

```
pre-commit install
```


### Mypy

There's a basic config for Mypy, but it doesn't pass yet.
