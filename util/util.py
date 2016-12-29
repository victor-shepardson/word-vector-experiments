import numpy as np
import libvictor as lv
from tqdm import tqdm_notebook
from tqdm import tqdm as tqdm_text
import pandas as pd
from collections import defaultdict

def make_pbar(verbose, notebook):
    def dummy(x, **kwargs):
        return x
    if verbose: return tqdm_notebook if notebook else tqdm_text
    return dummy

# seqs is a list of index sequences
# frequencies maps indices to word frequency in seqs.
#     if None, compute it from seqs
# TODO: allow context_size as argument to returned function when use_windows is False
def make_batch_generator(seqs, context_size, frequencies=None,
                         freq_exp=1,
                         use_windows=False, 
                         verbose=True, notebook=True):
    pbar = make_pbar(verbose, notebook)
    window_size = context_size+1
    if frequencies is None:
        #untested
        print('computing word frequencies')
        vocab_size = np.max([np.max(seq) for seq in seqs])+1
        frequencies = np.sum(np.bincounts(seq, minlength=vocab_size) for seq in pbar(seqs))
    else:
        freqencies = np.array(frequencies)
    if use_windows: # discard any sequences which are smaller than window size
        seqs = [seq for seq in seqs if len(seq)>=window_size]
    # get word frequencies and convert to sampling distribution
    print('precomputing cdfs')
    text_cdf = []
    loc_cdfs = []
    weights = np.power(frequencies, -freq_exp)
    for seq in pbar(seqs): 
        p = weights[seq]
        text_cdf.append(p.sum())
        if use_windows: # ends of sequences can't be selected as window centers
            p[:context_size//2]=0
            p[-context_size//2:]=0
        p/=p.sum() #normalize so p is a pmf
        p.cumsum(out=p)
        loc_cdfs.append(p)
    text_cdf = np.array(text_cdf)
    text_cdf/=text_cdf.sum()
    text_cdf.cumsum(out=text_cdf)
    def batch_generator(
        batch_size, n_batches,
        weight_words=True,
        weight_texts=True, #slow, should replace multinomal sampling w/ repeated cdf sampling
        seed=None):
        if not seed is None:
            np.random.seed(seed)
        for _ in range(n_batches):
            # first choose sequences to sample words from
            # this only weights window centers/inputs when sampling
            if weight_texts:
                seq_idxs = lv.sample_cdf(text_cdf, batch_size)
            else:
                seq_idxs = np.random.randint(len(seqs), size=batch_size)

            if use_windows:
                # then choose windows from those sequences
                if weight_words:
                    window_centers = np.array([lv.sample_cdf(loc_cdfs[i]) for i in seq_idxs])
                else:
                    window_centers = np.array([np.random.choice(
                                np.arange(context_size//2, len(seqs[s_idx])-context_size//2)
                            ) for s_idx in seq_idxs])
                windows = np.stack((seqs[s_idx][w_c-context_size//2:w_c+context_size//2+1] 
                                    for w_c, s_idx in zip(window_centers, seq_idxs)),0)

                # then rearrange windows into input, target pairs
                words = windows[:, context_size//2]#np.repeat(windows[:, context_size], window_size-1)
                context_idxs = [i for i in range(window_size) if i!=context_size//2]
                contexts = windows[:,context_idxs]
                inputs, targets = words, contexts
            else:
                #sample from whole sequences (i.e. sequences are sentences or paragraphs, not books)
                if weight_words:
                    idxs = np.array([seqs[i][lv.sample_cdf(loc_cdfs[i], window_size)] 
                                     for i in seq_idxs])
                else:
                    idxs = np.array([np.random.choice(seqs[i], window_size)
                                     for i in seq_idxs])
                # in this scheme, distinction b/t inputs and targets is arbitrary
                # true? is it a problem for inputs to appear in targets?
                inputs, targets = idxs[:,0], idxs[:, 1:]
                
            yield inputs, targets
    return batch_generator

# read BLESS word relationships file: https://sites.google.com/site/geometricalmodels/shared-evaluation
def read_BLESS(fname):
    df = pd.read_csv(fname, delim_whitespace=True, names=['concept', 'concept_class', 'relation', 'relatum'])
    f = lambda s: '-'.join(s.split('-')[:-1])
    df['concept'] = df.concept.map(f)
    df['relatum'] = df.relatum.map(f)
    return df

# convert the raw BLESS dataframe to a data structure organized by concept:
# { <concept>: {
#       'class': <class>,
#       <relation>: [<relatum>,...]
#   },
#  ...
# }
# discard concepts and relata not in vocab and report number
# vocab should have a fast "in" operator, e.g. be a dict with words as keys
def convert_BLESS(df, vocab):
    d = defaultdict(lambda: defaultdict(list))
    n_discarded = 0
    n_tests = 0
    for _, row in df.iterrows():
        c,r = row.concept, row.relatum
        if c in vocab and r in vocab:
            d[c]['class'] = row.concept_class
            d[c][row.relation].append(r)
        else:
            n_discarded+=1
        n_tests+=1
    print('rejected {} of {} tests with words missing from vocabulary'.format(n_discarded, n_tests))
    return d

def build_BLESS(fname, vocab):
    return convert_BLESS(read_BLESS(fname), vocab)

# given the bless data structure and an embedding mapping words to vectors,
# score each concept/relation
# assumes all concepts and relata are in vocabulary and embeddings are unit vectors
# scores are stored in bless[<concept>]['scores'][<relation>]
def test_BLESS(bless, word_embeddings, test_relations = ['attri', 'coord', 'event', 'hyper', 'mero', 'random-n', 'random-j', 'random-v'], score_fn = np.dot):
    for concept, d in bless.items():
        c_v = word_embeddings[concept]
        score = lambda relata: [score_fn(word_embeddings[r], c_v) for r in relata]
        d['scores'] = {relation:score(d[relation]) for relation in test_relations}

# compute single score for each relation over all concepts, relata
def aggregate_BLESS_scores(bless, test_relations = ['attri', 'coord', 'event', 'hyper', 'mero', 'random-n', 'random-j', 'random-v']):
    agg = defaultdict(list)
    for concept, d in bless.items():
        for relation in test_relations:
            agg[relation]+=d['scores'][relation]
    return {k:(np.mean(v, 0),np.std(v, 0)) for k,v in agg.items()}        