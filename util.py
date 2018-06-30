import sys
import functools
import logging
from contextlib import contextmanager
from time import time

from trec_car import read_data
from nltk import word_tokenize
import numpy as np
from tqdm import tqdm as tqdm_base


def tqdm(*args, **kwargs):
    return tqdm(*args, **kwargs, ncols=80)


@functools.lru_cache(maxsize=None) # no maximum
def get_logger(name, level='debug'):
    logFormatter = logging.Formatter('[%(asctime)s][%(name)s:%(lineno)s][%(levelname)s] - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logFormatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.propagate = False
    if level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'warning':
        logger.setLevel(logging.WARNING)
    elif level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif level == 'error':
        logger.setLevel(logging.ERROR)
    return Logger(logger)


def my_logger(level='debug'):
    """
    Builds and returns a logger from the caller's __name__
    """
    import inspect
    frame = inspect.stack()[1] # caller
    module = inspect.getmodule(frame[0])
    return get_logger(module.__name__, level)


class Logger(object):
    def __init__(self, logger):
        self.logger = logger

    def debug(self, text):
        self.logger.debug(text)

    def warn(self, text):
        self.logger.warn(text)

    @contextmanager
    def duration(self, message):
        t = time()
        self.logger.debug('[START] ' + message)
        yield
        self.logger.debug('[DONE] ' + message + ' [{:.4f}s]'.format(time()-t))

    def warn_first(self, message):
        def wrapper(wrapped):
            data = {'warned': False}
            def fn(*args, **kwargs):
                if not data['warned']:
                    self.logger.debug(message)
                    data['warned'] = True
                return wrapped(*args, **kwargs)
            return fn
        return wrapper

def read_run(file):
    if type(file) == list:
        for f in file:
            yield from read_run(f)
    elif hasattr(file, 'read'):
        for line in file:
            cols = line.split()
            if len(cols) == 2:
                yield cols
            elif len(cols) > 2:
                yield cols[0], cols[2] # TREC run and qrel format
            else:
                raise ValueError('Unexpected run file format')
    else:
        with open(file, 'rt') as f:
            yield from read_run(f)


def read_outlines(file):
    if type(file) == list:
        for f in file:
            yield from read_outlines(f)
    elif hasattr(file, 'read'):
        for outline in read_data.iter_outlines(file):
            for flat_headings in outline.flat_headings_list():
                qid = [outline.page_id]
                text = [outline.page_name]
                for heading in flat_headings:
                    qid.append(heading.headingId)
                    text.append(heading.heading)
                yield qid, text
    else:
        with open(file, 'rb') as f:
            yield from read_outlines(f)


def read_paragraphs(file):
    if type(file) == list:
        for f in file:
            yield from read_paragraphs(f)
    elif hasattr(file, 'read'):
        for paragraph in read_data.iter_paragraphs(file):
            yield paragraph.para_id, paragraph.get_text()
    else:
        with open(file, 'rb') as f:
            yield from read_paragraphs(f)


def tokenize(text):
    return [t.lower() for t in word_tokenize(text)]


def read_embeddings(file):
    if hasattr(file, 'read'):
        terms = []
        embeddings = []
        emb_len = None
        for line in file:
            cols = line.split()
            if emb_len is None and len(cols) == 2:
                emb_len = int(cols[1])
            else:
                if emb_len is None:
                    emb_len = len(cols) - 1
                terms.append(cols[0])
                embeddings.append(np.array([float(c) for c in cols[1:]]))
        embeddings = np.stack(embeddings)
        term_lookup = {t: i for i, t in enumerate(terms)}
        return {
            'tok': terms,
            'tok_lookup': term_lookup,
            'embeddings': embeddings,
            'missing_tok_lookup': {}
        }
    else:
        with open(file, 'rb') as f:
            return read_embeddings(f)
