import numpy as np
import hicstraw
from scipy.sparse import coo_matrix

def load_binsNum(hicfile, res):
    # get chrom lenght in .hic file; remove X, Y and MT chromosomes
    exclude_chroms = {'Y', 'MT', 'All', 'chrY'}
    hic = hicstraw.HiCFile(hicfile)
    chroms = hic.getChromosomes()
    chrom_bins = {}
    for chrom in chroms:
        if any(exclude in chrom.name for exclude in exclude_chroms):
            continue
        num_bins = (chrom.length + res) // res
        chrom_bins[chrom.name] = num_bins
    return chrom_bins

def normalizationMat(matrix, binsNum):
    # make the sum of rows' count equal to 1
    # count / rows_sum 
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    non_zero_mask = row_sums != 0
    matrix = matrix.tocoo()
    normed_count = np.zeros_like(matrix.data)
    for i in range(len(matrix.data)):
        row = matrix.row[i]
        if non_zero_mask[row]:
            normed_count[i] = matrix.data[i] / row_sums[row]
        else:
            normed_count[i] = matrix.data[i]
    normed_mat = coo_matrix((normed_count, (matrix.row, matrix.col)), shape=(binsNum, binsNum))
    return normed_mat

def constructSpaMat(result, binsNum, res):
    # Construct sparse matrix (A + A.T - A.diag)
    rows = []
    cols = []
    count = []
    for i in range(len(result)):
        rows.append(int(result[i].binX / res))
        cols.append(int(result[i].binY / res))
        count.append(result[i].counts)
    rows = np.array(rows)
    cols = np.array(cols)
    count = np.array(count)
    upper_mat = coo_matrix((count, (rows, cols)), shape=(binsNum, binsNum))
    lower_mat = coo_matrix((count, (cols, rows)), shape=(binsNum, binsNum))
    full_mat = upper_mat + lower_mat - coo_matrix((count[rows == cols], 
                                                 (rows[rows == cols],
                                                  cols[rows == cols])), shape=(binsNum, binsNum)) 
    return full_mat

def filter_matrix(mat):
    col_sum = np.sum(mat, axis=0)
    if np.any(col_sum < 0.1):
        return False
    else:
        return True

def dumpMatrix(chrom, binsNum, res, hicfile):
    """
    convert hic file to iced normalization sparse matrix
    :param chrom: chrom ID
    :param binsNum: the number of bin
    :param hicfile:  input hic file
    :param res: resolution for hic data
    """
    # load mat form .hic file
    result = hicstraw.straw('observed', 'KR', hicfile, str(chrom), str(chrom), 'BP', res)
    sp_mat = constructSpaMat(result, binsNum, res)
    normed_mat = normalizationMat(sp_mat, binsNum)
    return normed_mat.tocsr()