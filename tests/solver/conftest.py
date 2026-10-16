def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: solves the whole example suite with the full time budget "
        "(deselect with -m 'not slow')",
    )
