from bisect import bisect_left, bisect_right


def extract_range(L, range_min, range_max):
    '''
    Returns a list of objects l from list L s.t. range_min <= l[0] <= range_max.

    L *must* be sorted.
    '''
    range_min = (range_min, 0.0)
    range_max = (range_max, 0.0)
    left_idx = bisect_left(L, range_min)
    if left_idx == len(L):
        return

    right_idx = bisect_right(L, range_max)

    if right_idx == 0:
        return

    for ii in range(left_idx, right_idx):
        yield L[ii]

def closest(L, x):
    '''
    Returns the element l of L which minimizes abs(l[0], x).

    L *must* be sorted and nonempty.
    '''
    x = (x, 0.0)

    ii = bisect_right(L, x)

    candidates = []
    try:
        candidates = [L[ii]]
    except IndexError:
        pass

    if ii > 0:
        candidates.append(L[ii-1])

    x = x[0]

    return min(candidates, key = lambda tpl: abs(tpl[0] - x))
