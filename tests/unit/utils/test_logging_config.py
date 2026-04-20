import logging

from novelentitymatcher.utils import logging_config


def _reset_novelentitymatcher_logging():
    logger = logging.getLogger("novelentitymatcher")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logging_config._logging_configured = False
    return logger


def test_configure_logging_updates_formatter_when_verbose_changes():
    logger = _reset_novelentitymatcher_logging()

    logging_config.configure_logging(verbose=False)
    assert logger.handlers[0].formatter._fmt == "%(message)s"

    logging_config.configure_logging(verbose=True)
    assert logger.handlers[0].formatter._fmt == "[%(levelname)s] %(name)s: %(message)s"

    _reset_novelentitymatcher_logging()


def test_get_logger_preserves_package_namespace():
    assert (
        logging_config.get_logger("novelentitymatcher.core.matcher").name
        == "novelentitymatcher.core.matcher"
    )
    assert (
        logging_config.get_logger("core.matcher").name
        == "novelentitymatcher.core.matcher"
    )
