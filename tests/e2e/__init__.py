"""End-to-end smoke tests for the OJS + SKZ repository.

These tests are deliberately lightweight: they verify cross-cutting repository
invariants that any contributor change must preserve. They do not require
PHP, MySQL, or any service to be running, so they are safe to run in any CI
environment.

Run:

    pytest tests/e2e -v
"""
