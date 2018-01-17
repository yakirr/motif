from __future__ import print_function, division
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pwm', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--masks', nargs='+',
        default=['/n/groups/price/yakir/data/reference/hg19.bed'])

parser.add_argument('--relu-thresh', type=float, default=None)
parser.add_argument('--exp-base', type=float, default=2)

parser.add_argument('--pool', default='max')

parser.add_argument('--sigmoid-scaling', type=float, default=None)
parser.add_argument('--sigmoid-offset', type=float, default=None)

parser.add_argument('--sannot-chr', required=True, help='name of sannot to generate')
parser.add_argument('--bfile-chr',
        default='/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/'+\
                '1000G.EUR.QC.')
parser.add_argument('--refgenome-chr',
        default='/n/groups/price/yakir/data/reference/hg19_fasta/chr')
parser.add_argument('--chroms', nargs='+', default=range(1,23), type=int)
args = parser.parse_args()

print('initializing...')
import pandas as pd
import numpy as np
from pybedtools import BedTool
import gprim.dataset as gd
import gprim.annotation as ga
import gprim.genome as gg
import strutil as su; reload(su)

def process_mask(maskname, snps):
    print('processing mask', maskname)
    if maskname.endswith('.rsid'):
        df = pd.read_csv(maskname, header=None, names=['SNP'])
        snps = pd.merge(snps, df, on='SNP', how='inner')
    elif maskname.endswith('.bed'):
        mask = BedTool(maskname)
        snps = ga.bed_to_snps(mask, snps)
    elif maskname.endswith('.ensgid'):
        geneset = pd.read_csv(maskname, header=None, names=['ENSGID'])
        genesetbed = gg.geneset_to_bed(geneset, gene_col_name='ENSGID',
                windowsize=100000)
        snps = ga.bed_to_snps(genesetbed, snps)
    else:
        print('error:', maskname, 'ends with neither bed nor rsid nor esngid')
    print('after', maskname, len(snps), 'snps remaining')
    return snps

print('reading pwm')
pwm = su.read_pwm(args.pwm)

for c in args.chroms:
    print('chromosome', c)
    print('reading refpanel')
    refpanel = gd.Dataset(args.bfile_chr)
    snps = refpanel.bim_df(c)
    print(len(snps), 'snps read')

    #TODO: make it accept .rsid file as well
    print('intersecting with mask')
    for maskname in args.masks:
        snps = process_mask(maskname, snps)

    print('reading ref genome sequence')
    with open(args.refgenome_chr+str(c)+'.fa', 'r') as f:
        f.readline()
        refgenome = np.frombuffer(f.read().replace('\n','').upper(), dtype='S1')
    print('done. length is', len(refgenome))

    print('producing context collection for SNPs')
    contexts_a2, contexts_a1, pos_in_context = \
            su.snp_contexts_collection(refgenome, snps, pwm)
    print('done. length is', len(contexts_a2))

    if args.relu_thresh is not None:
        print('using relu with threshold of', args.relu_thresh)
        nl = su.relu(args.relu_thresh)
    else:
        print('using exp with base', args.exp_base)
        nl = su.expbasek(args.exp_base)

    if args.pool == 'max':
        pool=np.max
    elif args.pool == 'sum':
        pool=np.sum
    print('aggregating by', args.pool)

    if args.sigmoid_scaling is not None and args.sigmoid_offset is not None:
        classifier = lambda x: 1/(1+np.exp(-(args.sigmoid_scaling * x + args.sigmoid_offset)))
        print('using sigmoid with parameters', args.sigmoid_scaling, args.sigmoid_offset)
    else:
        classifier = lambda x: x
        print('no sigmoid')

    print('computing pwm scores for A2')
    snps['score_A2'] = classifier(
            su.score_contexts(contexts_a2, pwm, pool=pool, nonlinearity=nl))
    del contexts_a2
    print('computing pwm scores for A1')
    snps['score_A1'] = classifier(
            su.score_contexts(contexts_a1, pwm, pool=pool, nonlinearity=nl))
    del contexts_a1

    snps[args.name] = snps.score_A1 - snps.score_A2

    print('writing output with stem', args.sannot_chr)
    snps[['CHR','BP','SNP','A1','A2',args.name]].to_csv(
            args.sannot_chr+str(c)+'.sannot.gz',
            compression='gzip', sep='\t', index=False)
    print('success\n')
