from __future__ import print_function, division
import pandas as pd
import numpy as np

expbase2 = lambda x: np.power(2,x)
expbasek = lambda k:(lambda x: np.power(k,x))
relu = lambda thresh:(lambda x: (x-thresh)*(x-thresh > 0))

# TODO: check for snps positioned at boundaries of chromosomes
def snp_contexts_collection(genome, snps, pwm):
    if type(genome) is str:
        genome = np.frombuffer(genome, dtype='S1')

    snp_radius = len(pwm)-1

    # check that each SNP has one of its alleles represented in the reference sequence
    # if this fails, the SNPs are likely not coded according to the positive strand
    if not ((genome[snps.BP - 1] == snps.A1) | (genome[snps.BP - 1] == snps.A2)).all():
        print('ERROR, there was at least one SNP with neither allele in the',
                'reference sequence')

    # get position of each SNP in this context collection
    pos_in_context = np.arange(snp_radius, len(snps)*(2*snp_radius+1), 2*snp_radius+1)

    # produce the context collection with A2 and A1 respectively
    contexts_a2 = genome[np.concatenate([
        np.arange(p-snp_radius, p+snp_radius+1) for p in (snps.BP-1)
        ])]
    contexts_a2[pos_in_context] = snps.A2
    contexts_a1 = np.copy(contexts_a2)
    contexts_a1[pos_in_context] = snps.A1

    return contexts_a2, contexts_a1, pos_in_context

def one_hot(seq):
    if type(seq) is str:
        seq = np.frombuffer(seq, dtype='S1')
    res = np.zeros((len(seq), 4))
    for i, c in zip(range(4), ['A','C','G','T']):
        res[seq == c, i] = 1
    return res

def read_pwm(filepath):
    return pd.read_csv(filepath, sep='\t', skiprows=1, header=None, names=['A','C','G','T'])

def reverse_comp(pwm):
    res = pwm.copy()
    temp = res['A'].values.copy()
    res['A'] = res['T']
    res['T'] = temp
    temp = res['G'].values.copy()
    res['G'] = res['C']
    res['C'] = temp
    return res[::-1].reset_index(drop=True)

def diag_sum_(mat):
    return np.array([np.trace(mat.T, i) for i in range(len(mat))])
def diag_sum(mat):
    res = np.zeros(len(mat))
    res = mat[:,0].copy()
    for i in range(1, mat.shape[1]):
        res[:-i] += mat[i:,i]
    return res

def score_sequence(seq, pwm, nonlinearity=expbase2, output=True):
    rcpwm = reverse_comp(pwm)

    if output: print('\tgenerating one-hot encoding')
    ohs = one_hot(seq)

    if output: print('\tscoring forward pwm')
    matches = ohs.dot(pwm.T)
    probs = nonlinearity(diag_sum(matches))
    del matches

    if output: print('\tscoring rc pwm')
    rcmatches = ohs.dot(rcpwm.T)
    rcprobs = nonlinearity(diag_sum(rcmatches))
    del rcmatches

    return probs, rcprobs

# pool can be np.sum or np.max
def score_contexts(contexts, pwm, pool=np.max, nonlinearity=expbase2,
        blocklength=None, bleedlength=None, output=True):
    if blocklength is None: # then assume we're dealing with contexts around single snps
        blocklength = 2*len(pwm)-1
    if bleedlength is None:
        bleedlength = len(pwm)-1
    probs, rcprobs = score_sequence(contexts, pwm, nonlinearity=nonlinearity, output=output)
    agg_probs = pool(probs.reshape(-1, blocklength)[:,:-(len(pwm)-1)], axis=1)
    agg_rcprobs = pool(rcprobs.reshape(-1, blocklength)[:,:-(len(pwm)-1)], axis=1)
    return pool([agg_probs,agg_rcprobs], axis=0)
