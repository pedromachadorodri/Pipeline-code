#!/usr/bin/env python

"""
    epitopepredict analysis methods
    Created September 2013
    Copyright (C) Damien Farrell
    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""

from __future__ import absolute_import, print_function
import sys, os, shutil, string, types
import csv, glob, pickle, itertools
import math
import re
import time, random
from collections import OrderedDict
from operator import itemgetter
import numpy as np
import pandas as pd
import subprocess
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from . import base, sequtils, tepitope, utilities, peptutils

home = os.path.expanduser("~")
#fix paths!
genomespath = os.path.join(home, 'epitopedata')
datadir = os.path.join(home, 'testpredictions')

def get_AAcontent(df, colname, amino_acids=None):
    """Amino acid composition for dataframe with sequences"""
    return df.apply( lambda r: peptutils.get_AAfraction(str(r[colname]), amino_acids), 1)

def net_charge(df, colname):
    """Net peptide charge for dataframe with sequences"""
    return df.apply( lambda r: peptutils.net_charge(r[colname]),1)

def isoelectric_point(df):
    def getpi(seq):
        X = ProteinAnalysis(seq)
        return X.isoelectric_point()
    return df.apply( lambda r: getpi(r.peptide),1)

def peptide_properties(df, colname='peptide'):
    """Find hydrophobicity and net charge for peptides"""

    df['hydro'] = get_AAcontent(df, colname)
    df['net_charge'] = net_charge(df, colname)
    return df

def _center_nmer(x, n):
    """Get n-mer sequence for a peptide centered in the middle.
    This should be applied to a dataframe per row.
    Returns: a single sequence centred on the peptide

    """

    seq = x['translation']
    size = x.end-x.start
    l = int((size-n)/2.0)
    if size>n:
        if size%2 == 1: l1 = l+1
        else: l1=l
        start = x.start+l1
        end = x.end-l
    elif size<=n:
        if size%2 == 1: l1 = l-1
        else: l1=l
        start = x.start+l1
        end = x.end-l
    if start<=0:
        d=1-start
        start = start+d
        end = end+d
    seq = seq[start:end]
    #print(size, x.peptide, x.start, x.end, l, l1, start, end, seq, len(seq))
    return seq

def _split_nmer(x, n, key, margin=3, colname='peptide'):
    """Row based method to split a peptide in to multiple n-mers
        if it's too large. Returns a dataframe of 3 cols so should be
        applied using iterrows and then use concat.

    Args:
        x: row item
        n: length to split on
        key:
    """

    size = x.end-x.start
    m = margin

    if size <= n+m:
        seq = _center_nmer(x, n)
        return pd.DataFrame({colname: seq},index=[0])
    else:
        seq = x[key]
        o=size%n
        #print (size, o)
        if o<=margin:
            size=size-o
            seq = _center_nmer(x, size)
            #print (size)
        seqs=[]
        seq = x[key][x.start:x.end]
        if x.start==0: s=1
        else: s=0
        for i in range(s, size, n):
            if i+n>size:
                seqs.append(seq[o:o+n])
                #print (x.name,seq[x.start:x.start+n])
            else:
                seqs.append(seq[i:i+n])
                #print (seq[i:i+n])
        seqs = pd.Series(seqs)
        d = pd.DataFrame({colname:seqs})
        return d

def create_nmers(df, genome, length=20, seqkey='translation', key='nmer', how='split', margin=0):
    """Get n-mer peptide surrounding a set of sequences using the host
        protein sequence.

    Args:
        df: input dataframe with sequence name and start/end coordinates
        genome: genome dataframe with host sequences
        length: length of nmer to return
        seqkey: column name of sequence to be processed
        how: method to create the n-mer, split will try to split up
            the sequence into overlapping n-mes of length is larger than size
            center will center the peptide
        margin: do not split sequences below length+margin
    Returns:
        pandas Series with nmer values
    """

    cols = ['locus_tag','gene','translation']
    cols = list(set(cols) & set(genome.columns))
    #merge with genome dataframe but must keep index for re-merging
    if len(df)==0:
        return
    temp = df.merge(genome[cols],left_on='name',right_on='locus_tag',
                    how='left')#.set_index(df.index)
    #print (temp)
    if not 'end' in list(temp.columns):
        temp = base.get_coords(temp)
    #temp = base.get_coords(temp)

    if how == 'center':
        temp[key] = temp.apply( lambda r: _center_nmer(r, length), 1)
        res = temp
    elif how == 'split':
        res=[]
        for n,r in temp.iterrows():
            d = _split_nmer(r, length, seqkey, margin, key)
            d['index']=n
            d.set_index('index',inplace=True)
            res.append(d)
        res = pd.concat(res)
        #print (res)
        res = temp.merge(res,left_index=True,right_index=True,how='right').reset_index(drop=True)
        #print (res)
    res=res.drop([seqkey],1)
    return res

def get_overlaps(df1, df2, label='overlap', how='inside', kind='noise',mhc=''):
    """
    Overlaps for 2 sets of sequences where the positions in host sequence are stored
    in each dataframe as 'start' and 'end' columns

    Args:
        df1 : first set of sequences, a pandas dataframe with columns called
                start/end or pos
        df2: second set of sequences
        label: label for overlaps column
        how: may be 'any' or 'inside'
    Returns:
        First DataFrame with no. of overlaps stored in a new column
    """

    new=[]
    a = base.get_coords(df1)
    aa = list(df1.columns)
    # print(a)
    b = base.get_coords(df2)
    # bb=list(df2.columns)
    # print(bb)


    def overlap(x,y):
        f=0
        #print x['name'],x.peptide
        # print (x.start,x.end)
        for i,r in y.iterrows():
            # print(r)
            if how == 'inside':
                if ((x.start<=r.start) & (x.end>=r.end)):
                    f+=1
            elif how == 'any':
                if ((x.start<=r.start) & (x.end>r.start)) or \
                   ((x.start>=r.start) & (x.start<r.end)):
                    #t = abs(r.start-x.start)
                    #print (a, b)
                    f+=1
            #print (r.start,r.end, f)
        # print(comp)
        return f
    # Pedro - adição - def position - 220721
    def position(x,y):
        comp = []
        allele = []
        count=[]
        #print x['name'],x.peptide
        # print (x.start,x.end)
        for i,r in y.iterrows():
            # print(r)
            if how == 'inside':
                if ((x.start<=r.start) & (x.end>=r.end)):
                    comp.append((r.start,r.end))
            elif how == 'any':
                if ((x.start<=r.start) & (x.end>r.start)) or \
                    ((x.start>=r.start) & (x.start<r.end)):
                    #t = abs(r.start-x.start)
                    #print (a, b)
                    comp.append((r.start,r.end))
            #print (r.start,r.end, f)
        # print(comp)
        return comp

    bb=list(df2.columns)

    def more(x,y):
        bb=list(y.columns)
        allele = []
        count=[]
        #print x['name'],x.peptide
        # print (x.start,x.end)
        for i,r in y.iterrows():
            # print(r)
            if how == 'inside' and kind=='noise':
                if ((x.start<=r.start) & (x.end>r.start)) or \
                    ((x.start>=r.start) & (x.start<r.end)):
                    #t = abs(r.start-x.start)
                    #print (a, b)
                    allele.append(r[bb[5]])
                    count.append(r[bb[6]])
            elif how == 'any' and kind=='noise':
                if ((x.start<=r.start) & (x.end>r.start)) or \
                    ((x.start>=r.start) & (x.start<r.end)):
                    #t = abs(r.start-x.start)
                    #print (a, b)
                    allele.append(r[bb[5]])
                    count.append(r[bb[6]])
        return allele, count
    for n,df in a.groupby('name'):
        found = b[b.name==n]
        df[label] = df.apply(lambda r: overlap(r,found),axis=1)
        if kind=='mhc':
            df['position'] = df.apply(lambda r: position(r,found),axis=1)  # Pedro - adição - 220721
        elif kind=='noise':
            df['position'] = df.apply(lambda r: position(r,found),axis=1)  # Pedro - adição - 220721
            w = df.apply(lambda r: more(r,found),axis=1)  # Pedro - adição - 300721
            ale=[]
            ct=[]
            for i,j in w:
                ale.append(i)
                ct.append(j)
            df['allele'] = ale
            df['count'] = ct
        df['count']= df['count'].apply(lambda y: np.nan if len(y)==0 else y)
        df['allele']= df['allele'].apply(lambda y: np.nan if len(y)==0 else y)
        df['position']= df['position'].apply(lambda y: np.nan if len(y)==0 else y)
        df = df.replace(np.nan,'-')
            # print(w.iloc[0:0])
            # df['allele'] = w[0]
            # df['count'] = w[1]
            # df['count'] = df.apply(lambda r: more(r,found),axis=1)  # Pedro - adição - 300721
        new.append(df)
    result = pd.concat(new)

    #print ('%s with overlapping sequences' %len(result[result[label]>0]))
    return result

def get_orthologs(seq, db=None, expect=1, hitlist_size=400, equery=None,
                  email=''):
    """
    Fetch orthologous sequences using remote or local blast and return the records
    as a dataframe.

    Args:
        seq: sequence to blast
        db: the name of a local blast db
        expect: expect value
        equery: Entrez Gene Advanced Search options,
                (see http://www.ncbi.nlm.nih.gov/books/NBK3837/)
    Returns:
        blast results in a pandas dataframe
    """

    from Bio.Blast import NCBIXML,NCBIWWW
    from Bio import Entrez, SeqIO
    Entrez.email = email

    print ('running blast..')
    if db != None:
        #local blast
        SeqIO.write(SeqRecord(Seq(seq)), 'tempseq.faa', "fasta")
        sequtils.local_blast(db, 'tempseq.faa', output='my_blast.xml', maxseqs=100)
        result_handle = open("my_blast.xml")
        df = sequtils.get_blast_results(result_handle)
    else:
        try:
            result_handle = NCBIWWW.qblast("blastp", "nr", seq, expect=expect,
                                  hitlist_size=500,entrez_query=equery)
            time.sleep(2)
            savefile = open("my_blast.xml", "w")
            savefile.write(result_handle.read())
            savefile.close()
            result_handle = open("my_blast.xml")
            df = sequtils.get_blast_results(result_handle, local=False)
        except Exception as e:
            print ('blast timeout')
            return

    df = df.drop(['subj','positive','query_length','score'],1)
    df.drop_duplicates(subset=['definition','perc_ident'], inplace=True)
    df = df[df['perc_ident']!=100]
    return df

def alignment_to_dataframe(aln):
    alnrows = [[str(a.id),str(a.seq)] for a in aln]
    df = pd.DataFrame(alnrows,columns=['accession','seq'])
    return df

def align_blast_results(df, aln=None, idkey='accession', productkey='definition'):
    """
    Get gapped alignment from blast results using muscle aligner.
    """

    sequtils.dataframe_to_fasta(df, idkey=idkey, seqkey='sequence',
                        descrkey=productkey, outfile='blast_found.faa')
    aln = sequtils.muscle_alignment("blast_found.faa")
    alnrows = [[a.id,str(a.seq)] for a in aln]
    alndf = pd.DataFrame(alnrows,columns=['accession','seq'])
    #res = df.merge(alndf, left_index=True, right_index=True)
    res = df.merge(alndf, on=['accession'])
    res = res.drop('sequence',1)
    #get rid of duplicate hits
    #res.drop_duplicates(subset=['definition','seq'], inplace=True)
    res = res.sort_values(by='identity',ascending=False)
    print ('%s hits, %s filtered' %(len(df), len(res)))
    return res, aln

def get_species_name(s):
    """Find [species name] in blast result definition"""

    m = re.search(r"[^[]*\[([^]]*)\]", s)
    if m == None:
        return s
    return m.groups()[0]
# Importante
def find_conserved_sequences(seqs, alnrows):
    """
    Find if sub-sequences are conserved in given set of aligned sequences
    Args:
        seqs: a list of sequences to find
        alnrows: a dataframe of aligned protein sequences
    Returns:
        a pandas DataFrame of 1 or 0 values for each protein/search sequence
    """

    f=[]
    for i,a in alnrows.iterrows():
        sequence = a.seq
        found = [sequence.find(j) for j in seqs]
        # print(found)
        f.append(found)
    for n in ['species','accession','name']:
        if n in alnrows.columns:
            ind = alnrows[n]
            break
    s = pd.DataFrame(f,columns=seqs,index=ind)
    s = s.replace(-1,np.nan)
    s[s>0] = 1
    s = s.replace(np.nan,'-')
    # s = s.replace(1,'equal')
    return s
# Importante
def epitope_conservation(peptides, alnrows=None, unique_core=False,proteinseq=None, blastresult=None,
                         blastdb=None, perc_ident=50, equery='srcdb_refseq[Properties]'):
    """
    Find and visualise conserved peptides in a set of aligned sequences.
    Args:
        peptides: a list of peptides/epitopes
        alnrows: a dataframe of previously aligned sequences e.g. custom strains
        proteinseq: a sequence to blast and get an alignment for
        blastresult: a file of saved blast results in plain csv format
        equery: blast query string
    Returns:
        Matrix of 0 or 1 for conservation for each epitope/protein variant
    """

    import seaborn as sns
    sns.set_context("notebook", font_scale=1.4)

    if alnrows is None:
        if proteinseq == None:
            print ('protein sequence to blast or alignment required')
            return
        if blastresult == None or not os.path.exists(blastresult):
            blr = get_orthologs(proteinseq, equery=equery, blastdb=blastdb)
            if blr is None:
                return
            #if filename == None: filename = 'blast_%s.csv' %label
            blr.to_csv(blastresult)
        else:
            blr = pd.read_csv(blastresult, index_col=0)
        #blr = blr[blr.perc_ident>=perc_ident]
        alnrows, aln = align_blast_results(blr)
        #print (sequtils.formatAlignment(aln))

    if 'perc_ident' in alnrows.columns:
        alnrows = alnrows[alnrows.perc_ident>=perc_ident]
    if 'definition' in alnrows.columns:
        alnrows['species'] = alnrows.definition.apply(get_species_name)
    c = find_conserved_sequences(peptides, alnrows).T

    c = c.dropna(how='all')
    c = c.reset_index(level=0).rename(columns={'index':'epitope'}) # Pedro - adição - 180721
    # Pedro - adição - 180721
    if unique_core == True:
        c = c.drop_duplicates('epitope')
    else:
        pass
    # c = c.reindex_axis(c.sum(1).sort_values().index)
    if len(c) == 0:
        print ('no conserved epitopes in any sequence')
        return
    return c

# Pedro - adição - def loc_pos_epitope_conservation - 310821
def loc_pos_epitope_conservation(dataframe,epitope_conserved):
    
    pos=[]
    for e,j in epitope_conserved.iterrows():
    #     print(j.epitope)
        f=dataframe.query('peptide == @j.epitope')
    #     print(f)
        pos.append(f.pos.iloc[0])
    # print(pos)

    epitope_conserved['pos_epitope']=pos
    # filt.loc[:, ('pos')] = pos

    epitope_conserved=epitope_conserved.reindex(columns=['epitope', 'pos_epitope', 'similar_peptide','name','identity'])
    return epitope_conserved

# Pedro - adição - def filter_conservation - 180721
def filter_conservation(dataframe):
    frames = []
    for i in dataframe:
    #     print(i)
        find = dataframe[i].astype(str).str.contains('-',regex=True,na=False)
        frames.append(dataframe[find])    
    change = pd.concat(frames).drop_duplicates('epitope') # Pedro - adição - drop - 180721
    
    return change

# Pedro - adição - def search_conservation - 110821
def search_conservation(listpep,alnrows_withoutgap,dup_epitope=True):
    import difflib as dl
    e=[]
    for p in listpep:
        length=len(p)
        for ss,s in alnrows_withoutgap.iterrows():
            sequence = s.seq
            name = s.description
            alig = sequtils.pairwise_alignment(p,sequence)
            # for ç in alig:
            #     print(ç)
            pos = alig[0][0].find(p)
            frag = sequence[pos:(pos+length)]
            s = dl.SequenceMatcher(None, p, frag)
            if s.ratio() == 1:
    #             continue
                # e.append([p,"".join([frag+'<->'+name]),round(s.ratio(),4)])
                e.append([p,frag,name,round(s.ratio(),4)])
            else:
                new=[]
                jota=list(dl.ndiff(p, frag))
                sub=[r.replace(' ','') for r in jota]
    #             print(k)
                for u in sub:
                    if '+' in u:
                        dele=u.replace(u[0],'')
                        ch=dele.replace(dele[-1],'['+dele[-1]+']')
                        new.append(ch)
                    elif '-' in u:
                        p.replace(u[0],'')
                        continue
                    else:
                        new.append(u)
                une=[''.join(new[p:p + length]) for p in range(0, len(new), length)]
                une="".join(une)
                # e.append([p,"".join([une+'<->'+name]),round(s.ratio(),4)])
                e.append([p,une,name,round(s.ratio(),4)])
                
    df = pd.DataFrame(e, columns=['epitope','similar_peptide','name','identity'])
    if dup_epitope == True:
        df = df.drop_duplicates(subset=['epitope', 'similar_peptide','name'])
    else:
        pass
    return df

# Importante
def _region_query(P, eps, D):
	neighbour_pts = []
	for point in D:
		if abs(P - point)<=eps:
			neighbour_pts.append(point)
	return neighbour_pts
# Importante
def _expand_cluster(P, neighbour_pts, C, c_n, eps, min_pts, D, visited):

    flatten = lambda l: [i for sublist in l for i in sublist]
    C[c_n].append(P)
    for point in neighbour_pts:
        if point not in visited:
            visited.append(point)
            neighbour_pts_2 = _region_query(point, eps, D)
            if len(neighbour_pts_2) >= min_pts:
                neighbour_pts += neighbour_pts_2
        #print (point,C)
        if point not in flatten(C):
            C[c_n].append(point)
# Importante
def _dbscan(D, eps=5, minsize=2):
    """
    1D intervals using dbscan. Density-Based Spatial clustering.
    Finds core samples of high density and expands clusters from them.
    """
    from numpy.random import rand
    noise = []
    visited = []
    C = []
    c_n = -1
    for point in D:
        visited.append(point)
        neighbour_pts = _region_query(point, eps, D)
        # print(neighbour_pts)
        if len(neighbour_pts) < minsize:
            noise.append(point)
        else:
            C.append([])
            c_n+=1
            _expand_cluster(point, neighbour_pts, C, c_n,eps, minsize, D, visited)

    C = [i for i in C if len(i)>=minsize]
    # print(noise)
    # noi = [j for j in noise]
    # print(noi)
    #for cl in C:
    #    print (cl)
    return C

def dbscan(B=None, x=None, dist=7, minsize=4):
    """Use dbscan algorithm to cluster binder positions"""

    if B is not None:
        if len(B)==0:
            return
        x = sorted(B.pos.astype('int'))
    clusts = _dbscan(x, dist, minsize)
    # print (clusts)
    return clusts

def _dbscan_noise(D, eps=5, minsize=2):
    """
    1D intervals using dbscan. Density-Based Spatial clustering.
    Finds core samples of high density and expands clusters from them.
    """
    from numpy.random import rand
    noise = []
    for point in D:
        neighbour_pts = _region_query(point, eps, D)
        # print(neighbour_pts)
        if len(neighbour_pts) < minsize:
            noise.append([point])
        else:
            pass
    
    return noise

def dbscan_noise(B=None, x=None, dist=7, minsize=4):
    """Use dbscan algorithm to cluster binder positions"""

    if B is not None:
        if len(B)==0:
            return
        x = sorted(B.pos.astype('int'))
    noise = _dbscan_noise(x, dist, minsize)
    # print (noise)
    return noise

# Importante
def find_clusters(binders, dist=None, allmhc=True, min_binders=2, min_size=12, max_size=50,
                    genome=None, colname='peptide'):
    """
    Get clusters of binders for a set of binders.
    Args:
        binders: dataframe of binders
        dist: distance over which to apply clustering
        min_binders : minimum binders to be considered a cluster
        min_size: smallest cluster length to return
        max_size: largest cluster length to return
        colname: name for cluster sequence column
    Returns:
        a pandas Series with the new n-mers (may be longer than the initial dataframe
        if splitting)
    """
    # clou = []
    C=[]
    N=[]
    grps = list(binders.groupby('name'))
    length = binders.head(1).peptide.str.len().max()
    print (length)
    if dist == None:
        dist = length+1
        # print ('using dist for clusters: %s' %dist)
    for n,b in grps:
        if len(b)==0: continue
        clusts = dbscan(b,dist=dist,minsize=min_binders)
        noise = dbscan_noise(b,dist=dist,minsize=min_binders)
        print(clusts)
        # print(noise)
        if len(clusts) == 0:
            continue
        if allmhc == False:
            for c in clusts:
                    # print(c[-1])
                    # clou.append(c)
                    gaps = [c[i]-c[i-1] for i in range(1,len(c))]
                    C.append([n,min(c),max(c)+(length-1),len(c)]) # Pedro - substituição - length - 260521
            for g in noise:
                N.append([n,min(g),max(g)+(length-1)])
        elif allmhc == True:
            for c in clusts:
#                 print(row.pos, c[-1])
                fd = binders.loc[binders['pos'] == int(c[-1])]
                if any(fd.length == 15):
                    # print('tem 15 e 9 - ',c)
                    length = 15
                    # print(length)
                    gaps = [c[i]-c[i-1] for i in range(1,len(c))]
                    C.append([n,min(c),max(c)+(length-1),len(c)])
    #             elif all(fd.length == 15):
    #                 print('tem 15 - ',c)
    #                 length = 15
    #                 # print(length)
    #                 gaps = [c[i]-c[i-1] for i in range(1,len(c))]
    #                 C.append([n,min(c),max(c)+(length-1),len(c)])
                elif all(fd.length == 9):
                    # print('tem 9 - ',c)
                    length = 9
                    # print('9 >>',length)
                    gaps = [c[i]-c[i-1] for i in range(1,len(c))]
                    C.append([n,min(c),max(c)+(length-1),len(c)])
            for g in noise:
                fd_noise = binders.loc[binders['pos'] == int(g[-1])]
                if any(fd_noise.length == 15):
                    length = 15
                    # print(length)
                    N.append([n,min(g),max(g)+(length-1)])
                elif all(fd_noise.length == 15):
                    length = 9
    #                 print('9 >>',length)
                    N.append([n,min(g),max(g)+(length-1)])
            
    # print(N)
    # print(C)
    if len(C)==0:
        print ('no clusters')
        return pd.DataFrame()
    x = pd.DataFrame(C,columns=['name','start','end','epitopes'])
    xn = pd.DataFrame(N,columns=['name','start','end'])
    xn['length'] = (xn.end-xn.start)+1
    x['length'] = (x.end-x.start)+1
    x = x[x['length']>=min_size]
    x = x[x['length']<=max_size]
    xn=xn.sort_values(['start','length'],ascending=True)
    x=x.sort_values(['start','epitopes','length'],ascending=True) # Pedro - substituição - ['binders','length'],ascending=False - 260521

    if genome is not None:
        cols = ['locus_tag','translation']
        if 'gene' in genome.columns:
            cols.append('gene')
        x = x.merge(genome[cols],
                    left_on='name',right_on='locus_tag')
        x[colname] = x.apply(lambda r: r.translation[r.start:r.end], 1)
        x = x.drop(['locus_tag','translation'],1)
        x = x.drop_duplicates(colname)
        x = x.sort_values(by=['binders'],ascending=False)
        x = x.reset_index(drop=True)

    print ('%s clusters found in %s proteins' %(len(x),len(x.groupby('name'))))
    print ('%s noises found in %s proteins using dist = %d' %(len(xn),len(xn.groupby('name')),dist))
    #print
    return x, xn

# Pedro - adição - def format_allelecluster, filter_allelecluster, allelecluster - 230621

def format_allelecluster(Res):
    
    # listcompone = [i for i in Res if not all(j==0 for j in i)]
    # listcomptwo = [item.pop(0) for item in listcompone]
    formate = [[sorted(set(j), key=lambda x:j.index(x))] for j in Res]
    
    return formate

def filter_allelecluster(dataframecluster,dataframeconcat):
    # index = sorted(dataframecluster.index.astype('int'))
    # print(index)
    ale=[]
    # test=[]
    for j,jj in dataframecluster.iterrows():
        start = jj.start
        end=jj.end
        # test.append([start,end])
        allele=[]
        # print('j>>>',j,'start',start,'end',end)
        for i,ii in dataframeconcat.iterrows():
            # print('j>>>',j,'start',start,'end',end)
            # allele=[]
            pos = ii.pos
            length=ii.length
            # print('j>>>',j,'start',start,'end',end,'------',pos,length,((pos+length)-1))
            if ((start <= pos <= end) and (pos+length-1 <= end)) is True:
                # print(pos,length,((pos+length)-1))
                allele.append(ii.allele)
            else:
                pass
        # print('j>>>',j,'start',start,'end',end,'------',pos,length,((pos+length)-1))
        # print(allele)
        ale.append(allele)
    # print('ale-',ale)
    
    # ale = [name for sublist in ale for name in (sublist or [0])]
    # print(ale)
    
    # cond = '-' # ale[0]
    # res = [[cond]]
    # for item in ale[0:]:
    #     if item != cond:
    #         res[-1].append(item)
    #     else:
    #         cond = item
    #         res.append([cond])
    # print(res)
    
    formate = format_allelecluster(ale)
    # print('formate-',formate)
    
    return formate

def allelecluster(dfcluster,promiscue2,title1,title2):
    
    # formate = filter_allelecluster(dataframecluster,dataframeconcat)
    f=[]
    for g, h in promiscue2.groupby('name'):
        # print(h)
        for n, b in dfcluster.groupby('name'):
            # print(g,n)
            found = h[h.name==n]
            # print(found)
            if len(found) == 0:
                continue
            else:
                # print(b,found)
                formate = filter_allelecluster(b,found)
    #             print(finalmhct)
                f.extend(formate)
    # print(f)
    
    x = pd.DataFrame(f, columns=['col'])
    for i in range(len(x)):
        a = x.col[i]
        b=', '.join(a)
        x.loc[i] = b
    dfcluster[title1] = x
    
    c = [[len(j) for j in i] for i in f]
    y = pd.DataFrame(c, columns=['numb'])
    dfcluster[title2] = y

    
    return dfcluster

# Pedro - adição - 270521
def take_sequence(df,fastaframe):
    
    M=[]
    for g, h in df.groupby('name'):
        #     print(h)
            for n, b in fastaframe.groupby('locus_tag'):
                # print(g,n)
                sel = h[h.name==n]
                if len(sel) == 0:
                    continue
                else:
    #                 print(sel,b)
    #                 M=[]
                    for i in range(len(sel)):
                        # row = df.iloc[i]
                        sec = b.translation.apply(lambda x: x[int(sel.start.iloc[i]-1):int(sel.end.iloc[i])])
                        M.append(list(sec))
                    # print(C)

    dff = pd.DataFrame(M,columns=['cluster'])
    df = pd.concat([df, dff], axis=1)

    return df

def randomize_dataframe(df, seed=8):
    """Randomize order of dataframe"""

    np.random.seed(seed=seed)
    new = df.reset_index(drop=True)
    new = new.reindex(np.random.permutation(new.index))
    return new

def save_to_excel(df, n=94, filename='peptide_lists'):
    """
    Save a dataframe to excel with option of writing in chunks.
    """

    writer = pd.ExcelWriter('%s.xls' %filename)
    i=1
    chunks = df.groupby(np.arange(len(df)) // n)
    for g,c in chunks:
        c.to_excel(writer,'list'+str(i))
        i+=1
    writer.save()
    return

def tmhmm(fastafile=None, infile=None):
    """
    Get TMhmm predictions
    Args:
        fastafile: fasta input file to run
        infile: text file with tmhmm prediction output
    """

    if infile==None:
        tempfile = 'tmhmm_temp.txt'
        cmd = 'tmhmm %s > %s' %(fastafile,tempfile)
        infile = subprocess.check_output(cmd, shell=True, executable='/bin/bash')
    tmpred = pd.read_csv(infile, delim_whitespace=True, comment='#',
                      names=['locus_tag','v','status','start','end'])
    tmpred = tmpred.dropna()
    print ('tmhmm predictions for %s proteins' %len(tmpred.groupby('locus_tag')))
    lengths=[]
    for i,row in tmpred.iterrows():
        if row.status == 'TMhelix':
            lengths.append(row.end-row.start)
    #print np.mean(lengths), np.std(lengths)
    return tmpred

def signalP(infile=None,genome=None):
    """Get signal peptide predictions"""

    if genome != None:
        seqfile = Genome.genbank2Fasta(genome)
        tempfile = 'signalp_temp.txt'
        cmd = 'signalp -t gram+ -f short %s > %s' %(seqfile,tempfile)
        infile = subprocess.check_output(cmd, shell=True, executable='/bin/bash')
    sp = pd.read_csv(infile,delim_whitespace=True,comment='#',skiprows=2,
                      names=['locus_tag','Cmax','cpos','Ymax','ypos','Smax',
                            'spos','Smean','D','SP','Dmaxcut','net'])
    #print sp[sp.SP=='Y']
    return sp

def get_seqdepot(seq):
    """Fetch seqdepot annotation for sequence"""

    from epitopepredict import seqdepot
    reload(seqdepot)
    sd = seqdepot.new()
    aseqid = sd.aseqIdFromSequence(seq)
    try:
        result = sd.findOne(aseqid)
    except Exception as e:
        print (e)
        result=None
    return result

def prediction_coverage(expdata, binders, key='sequence', perc=50, verbose=False):
    """
    Determine hit rate of predictions in experimental data
    by finding how many top peptides are needed to cover % positives
    Args:
        expdata: dataframe of experimental data with peptide sequence and name column
        binders: dataframe of ranked binders created from predictor
        key: column name in expdata for sequence
    Returns:
        fraction of predicted binders required to find perc total response
    """

    def getcoverage(data, peptides, key):
        #get coverage for single sequence
        target = math.ceil(len(data)*perc/100.0)
        if verbose == True:
            print (len(data), target)
        #print data[key]
        #print peptides[peptides.isin(data[key])]
        found=[]
        count=0
        for p in peptides:
            for i,r in data.iterrows():
                #print p, r[key]
                if r[key] in found:
                    continue
                if r[key].find(p)!=-1 or p.find(r[key])!=-1:
                    found.append(r[key])
                    if verbose == True:
                        print (count, p, r[key])
                    continue
            count+=1
            if len(found) >= target:
                if verbose == True:
                    print (count, target)
                    print ('--------------')
                return count
        if verbose == True:
            print ('not all sequences found', count, target)
        return count

    total = 0
    for name, data in expdata.groupby('name'):
        peptides = binders[binders.name==name].peptide
        if len(peptides) == 0:
            continue
        if verbose == True: print (name)
        #print (binders[binders.name==name][:5])
        c = getcoverage(data, peptides, key)
        total += c

    #print (total, total/float(len(binders))*100)
    return round(total/float(len(binders))*100,2)

def test_features():
    """test feature handling"""

    fname = os.path.join(datadir,'MTB-H37Rv.gb')
    df = sequtils.genbank2Dataframe(fname, cds=True)
    df = df.set_index('locus_tag')
    keys = df.index
    name='Rv0011c'
    row = df.ix[name]
    seq = row.translation
    prod = row['product']
    rec = SeqRecord(Seq(seq),id=name,description=prod)
    fastafmt = rec.format("fasta")
    print (fastafmt)
    print (row.to_dict())
    ind = keys.get_loc(name)
    previous = keys[ind-1]
    if ind<len(keys)-1:
        next = keys[ind+1]
    else:
        next=None
    return

def testrun(gname):

    method = 'tepitope'#'iedbmhc1'#'netmhciipan'
    path='test'
    gfile = os.path.join(genomespath,'%s.gb' %gname)
    df = sequtils.genbank2Dataframe(gfile, cds=True)
    #names = list(df.locus_tag[:1])
    names=['VP24']
    alleles1 = ["HLA-A*02:02", "HLA-A*11:01", "HLA-A*32:07", "HLA-B*15:17", "HLA-B*51:01",
              "HLA-C*04:01", "HLA-E*01:03"]
    alleles2 = ["HLA-DRB1*0101", "HLA-DRB1*0305", "HLA-DRB1*0812", "HLA-DRB1*1196", "HLA-DRB1*1346",
            "HLA-DRB1*1455", "HLA-DRB1*1457", "HLA-DRB1*1612", "HLA-DRB4*0107", "HLA-DRB5*0203"]
    P = base.getPredictor(method)
    P.iedbmethod='IEDB_recommended' #'netmhcpan'
    P.predictProteins(df,length=11,alleles=alleles2,names=names,
                        save=True, path=path)
    f = os.path.join('test', names[0]+'.mpk')
    df = pd.read_msgpack(f)
    P.data=df
    #b = P.get_binders(data=df)
    #print b[:20]
    base.getScoreDistributions(method, path)
    return

def test_conservation(label,gname):
    """Conservation analysis"""

    tag='VP24'
    pd.set_option('max_colwidth', 800)
    gfile = os.path.join(genomespath,'%s.gb' %gname)
    g = sequtils.genbank2Dataframe(gfile, cds=True)
    res = g[g['locus_tag']==tag]
    seq = res.translation.head(1).squeeze()
    print (seq)
    #alnrows = getOrthologs(seq)
    #alnrows.to_csv('blast_%s.csv' %tag)
    alnrows = pd.read_csv('blast_%s.csv' %tag,index_col=0)
    alnrows.drop_duplicates(subset=['accession'], inplace=True)
    alnrows = alnrows[alnrows['perc_ident']>=60]
    seqs=[SeqRecord(Seq(a.sequence),a.accession) for i,a in alnrows.iterrows()]
    print (seqs[:2])
    sequtils.distanceTree(seqs=seqs)#,ref=seqs[0])
    #sequtils.ETETree(seqs, ref, metric)
    #df = sequtils.getFastaProteins("blast_found.faa",idindex=3)
    '''method='tepitope'
    P = base.getPredictor(method)
    P.predictSequences(df,seqkey='sequence')
    b = P.get_binders()'''
    return
# importante
def find_conserved_peptide(peptide, recs):
    """Find sequences where a peptide is conserved"""

    f=[]
    for i,a in recs.iterrows():
        seq = a.sequence.replace('-','')
        found = seq.find(peptide)
        f.append(found)
    s = pd.DataFrame(f,columns=['found'],index=recs.accession)
    s = s.replace(-1,np.nan)
    #print s
    res = s.count()
    return s

def test():
    gname = 'ebolavirus'
    label = 'test'

    testrun(gname)
    #testBcell(gname)
    #testgenomeanalysis(label,gname,method)
    #testconservation(label,gname)
    #testFeatures()
    return

if __name__ == '__main__':
    pd.set_option('display.width', 600)
    test()
