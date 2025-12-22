# Documentation notes

This folder contains auxiliary mkdocs files. The main documentation pages are located under `docs/` (top-level) and are listed in `mkdocs.yml`.

Common commands:

```bash
mkdocs serve   # Run a local documentation server with live reload
mkdocs build   # Build the static site into the "site/" folder
```

To preview style changes, update CSS in `docs/assets/stylesheets/custom-theme.css` and run `mkdocs serve`.

If you want to extend the docs with API-generated content, configure `mkdocstrings` in `mkdocs.yml` and add docstring-compatible modules under `optiflowx/`.
