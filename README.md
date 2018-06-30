# PACRR-CAR

Utilities for running the [PACRR neural IR model](https://github.com/khui/copacrr)
for [Complex Answer Retrieval](http://trec-car.cs.unh.edu/).

So far, there's just similarity matrix generation (to [HDF5](https://www.h5py.org/) file).

## Setup

```
pip install -r requirements.txt
```

Python version: 3.6

## Usage

```
python generate_simmats.py --run [qrels] --outlines [outlines.cbor] --embeddings [embed] --paragraphs [paragraphcorpus.cbor]
```

Outputs to `simmats.hdf5`. To configure, use `--output`.

The `--run` argument is a TREC qrels or run file that indicates which simmats to
generate. The `--outlines` file is a cbor file that indicates the text of the
queries. The `--paragraphs` file is a cbor file that contains the text of the
paragraphs. Both can be found at http://trec-car.cs.unh.edu/datareleases/.

This will run on as many CPUs as are available on the machine. To configure this
value, use the `--pool` option.

On 24 CPUs, it takes about 30min to generate simmats for automatic in 1 fold
(data version 1.5). It's pretty I/O heavy right now, so there's probably a
way to make it faster. It takes a while just to read through the paragraphs
file itself, though.

### Samle

```
python generate_simmats.py --run car-train/train.fold0.cbor.hierarchical.qrels --outlines car-train/train.fold0.cbor.outlines --embeddings glove.6B.50d.txt --paragraphs car-paragraphcorpus/paragraphcorpus.cbor
[2018-06-30 16:52:02,509][__main__:59][DEBUG] - [START] reading run files
1054369it [00:03, 272597.76it/s]
[2018-06-30 16:52:06,377][__main__:51][DEBUG] - found 1054369 pairs, 436851 qids, 1030775 docids
[2018-06-30 16:52:06,377][__main__:61][DEBUG] - [DONE] reading run files [3.8686s]
[2018-06-30 16:52:06,377][__main__:59][DEBUG] - [START] reading outlines
408137it [01:19, 5149.98it/s]
[2018-06-30 16:53:25,628][__main__:51][DEBUG] - found 408004 qids, 266457 headings
[2018-06-30 16:53:25,628][__main__:61][DEBUG] - [DONE] reading outlines [79.2507s]
[2018-06-30 16:53:25,785][__main__:54][WARNING] - missing outlines for 28847 qid(s), e.g. 9/11%20Truth%20movement/History
[2018-06-30 16:53:25,785][__main__:59][DEBUG] - [START] cleaning up missing qids
[2018-06-30 16:53:25,842][__main__:61][DEBUG] - [DONE] cleaning up missing qids [0.0573s]
[2018-06-30 16:53:25,842][__main__:59][DEBUG] - [START] loading embeddings
[2018-06-30 16:53:33,795][__main__:61][DEBUG] - [DONE] loading embeddings [7.9529s]
[2018-06-30 16:53:33,992][__main__:54][WARNING] - missing 35662 token(s) in embeddings, e.g. hauptstadt. These will be treated as binary matches. Consider retraining embeddings.
[2018-06-30 16:53:34,004][__main__:59][DEBUG] - [START] generating simmats
[2018-06-30 16:53:34,019][__main__:51][DEBUG] - pool process started
...
[2018-06-30 16:53:34,337][__main__:51][DEBUG] - pool process started
100%|██████████████████████████████| 1030775/1030775 [32:53<00:00, 522.19para/s]
[2018-06-30 17:26:28,273][__main__:61][DEBUG] - [DONE] generating simmats [1974.2693s]
[2018-06-30 17:26:28,273][__main__:51][DEBUG] - done!
```

File size: 5.2G

## Details

The output file is organized as follows:

```
[heading_or_page_id]/[paragraphid] - q x d similarity matrix
```

When building the full simmat for a query, find all components of the query
(i.e., page_id, and heading ids), and concat the matrices along axis 0.

Terms that are not found in the embeddings file are binary matched (i.e,
a similarity score of 1 if exact match, otherwise 0). It's best to retrain
the embeddings using all available data so there are no missing terms.
