# Publishing to PyPI

This project uses PyPI Trusted Publishing with GitHub Actions and OpenID Connect. No long-lived PyPI API token is stored in GitHub.

## Trusted Publisher identity

Configure these exact values on PyPI:

| Field | Value |
| --- | --- |
| PyPI project name | `markov-entropy` |
| GitHub owner | `jiangnan030-del` |
| GitHub repository | `markov-entropy` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

For the first publication, create a pending publisher from the PyPI account publishing page. If the project already exists, add the same identity from the project's **Manage → Publishing** page.

A pending publisher reserves no project name. Complete the first trusted publication promptly after configuring it.

## GitHub environment

Create an environment named `pypi` under **Repository settings → Environments**. Recommended protection:

- restrict deployment branches and tags to release tags;
- add a required reviewer for production publication;
- do not add a `PYPI_TOKEN` secret.

The environment name must exactly match the trusted publisher configuration.

## Automated workflow

`.github/workflows/publish.yml`:

1. checks out the Git tag being published;
2. verifies that `vX.Y.Z` matches `project.version`;
3. builds the wheel and source distribution with uv;
4. validates both distributions with Twine;
5. passes the files to an isolated publish job;
6. obtains a short-lived PyPI credential through GitHub OIDC;
7. publishes with digital attestations.

The publish job alone receives `id-token: write` permission.

## First publication of v0.3.0

The v0.3.0 GitHub Release predates the publishing workflow, so it will not trigger automatically. After merging the trusted-publishing PR and configuring PyPI:

1. Open **Actions → Publish to PyPI**.
2. Select **Run workflow** on `main`.
3. Enter `v0.3.0` as the tag.
4. Approve the `pypi` environment deployment if protection rules require it.
5. Verify `https://pypi.org/project/markov-entropy/0.3.0/`.

Do not recreate or move the `v0.3.0` tag. The workflow checks out the existing immutable release tag.

## Future releases

For future versions:

1. update `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md`, and `uv.lock`;
2. run the full validation suite and build checks;
3. merge the release PR;
4. create and publish a GitHub Release with a matching `vX.Y.Z` tag;
5. the PyPI workflow starts automatically;
6. verify the PyPI project page and installation in a clean environment.

## Recovery and security

- Never paste PyPI passwords or API tokens into issues, commits, logs, or chat.
- A failed OIDC match usually means the owner, repository, workflow filename, or environment differs from PyPI's configuration.
- Publishing an existing version fails by design; PyPI files are immutable.
- Do not enable `skip-existing` for production releases because it can hide an incomplete or inconsistent release.
