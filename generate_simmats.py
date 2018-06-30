from collections import defaultdict
import h5py
from tqdm import tqdm as tqdm_base
import itertools
from multiprocessing import Pool
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import util

logger = util.my_logger()


def tqdm(*args, **kwargs):
    return tqdm_base(*args, **kwargs, ncols=80)


def build_workload(runs):
    docids_by_qid = defaultdict(set)
    qids_by_docid = defaultdict(set)
    count = 0
    with logger.duration('reading run files'):
        for qid, docid in tqdm(util.read_run(runs)):
            docids_by_qid[qid].add(docid)
            qids_by_docid[docid].add(qid)
            count += 1
        logger.debug('found {} pairs, {} qids, {} docids'.format(count, len(docids_by_qid), len(qids_by_docid)))
    return docids_by_qid, qids_by_docid


def build_qid_map(outlines):
    qid_map = {}
    heading_map = {}
    with logger.duration('reading outlines'):
        for outline_qid, outline_text in tqdm(util.read_outlines(outlines)):
            qid_map['/'.join(outline_qid)] = outline_qid
            for heading, text in zip(outline_qid, outline_text):
                heading_map[heading] = util.tokenize(text)
        logger.debug('found {} qids, {} headings'.format(len(qid_map), len(heading_map)))
    return qid_map, heading_map


def generate_tasks(paras, run_qids_by_docid, qid_map, heading_map):
    for docid, doctext in util.read_paragraphs(paras):
        if docid in run_qids_by_docid:
            work = {}
            for qid in run_qids_by_docid[docid]:
                for heading_id in qid_map[qid]:
                    work[heading_id] = heading_map[heading_id]
            yield docid, doctext, work


pool_embeddings = None


def pool_make_reps(toks):
    global pool_embeddings
    emb = np.zeros((len(toks), pool_embeddings['embeddings'].shape[1]), dtype=float)
    replace = np.full((len(toks),), -1, dtype=int)
    for i, tok in enumerate(toks):
        if tok in pool_embeddings['tok_lookup']:
            emb[i,:] = pool_embeddings['embeddings'][pool_embeddings['tok_lookup'][tok]]
        elif tok in pool_embeddings['missing_tok_lookup']:
            replace[i] = pool_embeddings['missing_tok_lookup'][tok]
    return emb, replace


def pool_generate_simmats(data):
    docid, doctext, queries_by_qid = data
    simmats = {}
    doc_tok = util.tokenize(doctext)
    doc_emb, doc_replace = pool_make_reps(doc_tok)
    for qid, qtoks in queries_by_qid.items():
        q_emb, q_replace = pool_make_reps(qtoks)
        simmat = cosine_similarity(q_emb, doc_emb)
        q_replace = np.expand_dims(q_replace, axis=1)
        q_replace = np.repeat(q_replace, len(doc_tok), axis=1)
        doc_replace_tmp = np.expand_dims(doc_replace, axis=0)
        doc_replace_tmp = np.repeat(doc_replace_tmp, len(qtoks), axis=0)
        mask = np.logical_and(np.logical_and(q_replace == doc_replace_tmp, q_replace != -1), doc_replace_tmp != -1)
        simmat[mask] = 1.
        simmats[qid] = simmat
    return docid, simmats


def pool_init(emb):
    global pool_embeddings
    pool_embeddings = emb
    logger.debug('pool process started')


def main():
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser('CAR similarity matrix generator')
    parser.add_argument('--run', help='TREC run/qrels file(s) to generate similarity matrices for (qid=cols[0], docid=cols[2])', nargs='+', type=FileType('rt'))
    parser.add_argument('--outlines', help='TREC-CAR outline file(s) (cbor)', nargs='+', type=FileType('rb'))
    parser.add_argument('--paragraphs', help='TREC-CAR paragraph file(s) (cbor)', nargs='+', type=FileType('rb'))
    parser.add_argument('--embeddings', help='Word embedding file.', type=FileType('rt'))
    parser.add_argument('--output', help='Output file (HDF5). If exists, appends. (default: simmats.hdf5)', type=h5py.File, default='simmats.hdf5')
    parser.add_argument('--pool', help='Number of processes to use for simmat generation (default: number of processors on machine)', default=None, type=int)
    args = parser.parse_args()

    run_docids_by_qid, run_qids_by_docid = build_workload(args.run)
    qid_map, heading_map = build_qid_map(args.outlines)

    missing_qids = run_docids_by_qid.keys() - qid_map.keys()
    if missing_qids:
        example_qid = next(iter(missing_qids))
        logger.warn('missing outlines for {} qid(s), e.g. {}'.format(len(missing_qids), example_qid))
        with logger.duration('cleaning up missing qids'):
            for missing_qid in missing_qids:
                for docid in run_docids_by_qid[missing_qid]:
                    run_qids_by_docid[docid].remove(missing_qid)
                del run_docids_by_qid[missing_qid]

    with logger.duration('loading embeddings'):
        embeddings = util.read_embeddings(args.embeddings)

    all_tokens = set().union(*heading_map.values())
    missing_tokens = all_tokens - embeddings['tok_lookup'].keys()
    if missing_tokens:
        example_token = next(iter(missing_tokens))
        logger.warn('missing {} token(s) in embeddings, e.g. {}. These will be treated as binary matches. Consider retraining embeddings.'.format(len(missing_tokens), example_token))
        for i, missing_token in enumerate(missing_tokens):
            embeddings['missing_tok_lookup'][missing_token] = i + len(embeddings['tok_lookup'])

    with logger.duration('generating simmats'):
        pool = Pool(args.pool, pool_init, (embeddings,))
        with tqdm(unit='para', total=len(run_qids_by_docid), smoothing=0.1) as pbar:
            task_generator = generate_tasks(args.paragraphs, run_qids_by_docid, qid_map, heading_map)
            for docid, mat_by_qid in pool.imap_unordered(pool_generate_simmats, task_generator, chunksize=10):
                for qid, mat in mat_by_qid.items():
                    key = '{}/{}'.format(qid, docid)
                    if key in args.output:
                        del args.output[key]
                    args.output[key] = mat
                del run_qids_by_docid[docid]
                pbar.update(1)

    if run_qids_by_docid:
        example_para = next(iter(run_qids_by_docid))
        logger.warn('missing {} paragraph(s), e.g., {}'.format(len(run_qids_by_docid), example_para))

    logger.debug('done!')

if __name__=='__main__':
    main()
