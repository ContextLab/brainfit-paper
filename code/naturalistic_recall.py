# noinspection PyPackageRequirements
import datawrangler as dw

import math

import numpy as np
import pandas as pd
from pycircstat.tests import rayleigh
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import hypertools as hyp

import os
import pickle
import warnings

from loader import DATA_DIR, get_video_and_recall_trajectories, get_formatted_data, sliding_windows

STEP = 0.9
XGRID_SCALE = [-10, 25]
YGRID_SCALE = [-17.5, 19.5]


# noinspection PyShadowingNames
def add_arrows(axes, x, y, *, aspace=None, head_width=None, **kwargs):
    # spacing of arrows
    if aspace is None:
        aspace = .05 * np.max(np.abs([*XGRID_SCALE, *YGRID_SCALE]))
    if head_width is None:
        head_width = aspace / 3
    # distance spanned between pairs of points
    r = [0]
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        r.append(np.sqrt(dx * dx + dy * dy))

    r = np.array(r)
    # cumulative sum of r, used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())
    # will hold tuple(x, y, theta) for each arrow
    arrow_data = []
    # current point on walk along data
    arrow_pos = 0
    rcount = 1
    while arrow_pos < r.sum():
        x1, x2 = x[rcount - 1], x[rcount]
        y1, y2 = y[rcount - 1], y[rcount]
        da = arrow_pos - rtot[rcount]
        theta = np.arctan2((x2 - x1), (y2 - y1))
        ax = np.sin(theta) * da + x1
        ay = np.cos(theta) * da + y1
        arrow_data.append((ax, ay, theta))
        arrow_pos += aspace
        while arrow_pos > rtot[rcount + 1]:
            rcount += 1
            if arrow_pos > rtot[-1]:
                break

    for ax, ay, theta in arrow_data:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit
        axes.arrow(ax,
                   ay,
                   np.sin(theta) * aspace / 10,
                   np.cos(theta) * aspace / 10,
                   head_width=head_width,
                   **kwargs)


class Point:
    def __init__(self, coord=None):
        self.coord = np.array(coord)


class LineSegment:
    def __init__(self, p1=None, p2=None):
        if not isinstance(p1, Point):
            p1 = Point(p1)
        if not isinstance(p2, Point):
            p2 = Point(p2)

        self.p1 = p1
        self.p2 = p2
        self.vec = self.p2.coord - self.p1.coord
        self.norm = self.vec / np.linalg.norm(self.vec)

    @property
    def angle(self):
        p1 = np.zeros_like(self.get_p1())
        p2 = np.zeros_like(self.get_p1())
        p2[0] = 1
        ref = LineSegment(p1, p2)
        return self.angle_with(ref)

    def get_p1(self):
        return self.p1.coord

    def get_p2(self):
        return self.p2.coord

    def intersects(self, z):
        if isinstance(z, Circle):
            return _seg_intersect_circle(self, z)
        elif isinstance(x, Rectangle):
            return _seg_intersect_rect(self, z)

    def angle_with(self, ref):
        assert isinstance(ref, LineSegment)
        v0 = ref.vec
        v1 = self.vec
        angle = np.arccos(v0.dot(v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if self.vec[1] < 0:
            angle = (2 * np.pi) - angle

        return angle


class Circle:
    def __init__(self, center=None, r=None):
        self.center = np.array(center)
        self.r = r

    def get_center(self):
        return self.center

    def get_radius(self):
        return self.r


class Rectangle:
    # noinspection PyShadowingNames
    def __init__(self, x=None, y=None, w=None):
        self.c0 = x - w
        self.c1 = y - w
        self.c2 = x + w
        self.c3 = y + w


def _seg_intersect_circle(ls, circ):
    Q = circ.get_center()
    r = circ.get_radius()
    P1 = ls.get_p1()
    V = ls.get_p2() - P1

    a = V.dot(V)
    b = 2 * V.dot(P1 - Q)
    c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r ** 2

    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return False

    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
        return False

    return True


def _seg_intersect_rect(ls, r):
    # find min/max X for the segment
    minX = min(ls.p1.x, ls.p2.x)
    maxX = max(ls.p1.x, ls.p2.x)

    # find the intersection of the segment's and rectangle's x-projections
    if maxX > r.c2:
        maxX = r.c2
    if minX < r.c0:
        minX = r.c0

    if minX > maxX:
        return False

    minY = ls.p1.y
    maxY = ls.p2.y

    dx = ls.p2.x - ls.p1.x

    if abs(dx) > .0000001:
        a = (ls.p2.y - ls.p1.y) / dx
        b = ls.p1.y - a * ls.p1.x
        minY = a * minX + b
        maxY = a * maxX + b

    if minY > maxY:
        tmp = maxY
        maxY = minY
        minY = tmp

    # find the intersection of the segment's and rectangle's y-projections
    if maxY > r.c3:
        maxY = r.c3
    if minY < r.c1:
        minY = r.c1

    # if Y-projections do not intersect return false
    if minY > maxY:
        return False
    else:
        return True


def compute_coord(xi, yi, w, seglist, kind='rectangle'):
    if kind == 'rectangle':
        z = Rectangle(x=xi, y=yi, w=w)
    elif kind == 'circle':
        z = Circle(center=[xi, yi], r=w)

    segs = list(filter(lambda s: s.intersects(z), seglist))
    c = len(segs)
    if c > 1:
        u, v = np.array([seg.norm for seg in segs]).mean(0)
        rads = np.array([seg.angle for seg in segs])
        p, z = rayleigh(rads)
    else:
        u = 0
        v = 0
        p = 1
    return u, v, p, c


# noinspection PyShadowingNames
def get_average_recall_events(video, recalls):
    avg_rec = np.zeros_like(video)
    counts = np.zeros(video.shape[0])
    for x in recalls:
        # drop nans
        r = x[~np.isnan(np.sum(x, axis=1)), :]

        if (r.shape[0] == 0) or (r.shape[1] == 0):
            continue

        participant_avg = np.zeros_like(video)
        participant_counts = np.zeros(video.shape[0])

        corrs = 1 - cdist(r, video, metric='correlation')
        for i in range(r.shape[0]):
            best_match = np.argmax(corrs[i, :])
            participant_avg[best_match, :] += r[i, :]
            participant_counts[best_match] += 1

        for i in range(participant_avg.shape[0]):
            if participant_counts[i] > 0:
                participant_avg[i, :] /= participant_counts[i]
                participant_counts[i] = 1

        avg_rec += participant_avg
        counts += participant_counts

    for i in range(avg_rec.shape[0]):
        if counts[i] > 0:
            avg_rec[i, :] /= counts[i]

    return avg_rec[counts > 0 , :]


def _segments_intersect2d(a1, b1, a2, b2):
    s1 = b1 - a1
    s2 = b2 - a2

    s = (-s1[1] * (a1[0] - a2[0]) + s1[0] * (a1[1] - a2[1])) / (-s2[0] * s1[1]
                                                                + s1[0] * s2[1])
    t = (s2[0] * (a1[1] - a2[1]) - s2[1] * (a1[0] - a2[0])) / (-s2[0] * s1[1]
                                                               + s1[0] * s2[1])

    if (s >= 0) and (s <= 1) and (t >= 0) and (t <= 1):
        return True
    else:
        return False


# noinspection PyShadowingNames
def n_intersections(x):
    intersections = 0
    for i in np.arange(x.shape[0] - 1):
        a1 = x[i, :]
        b1 = x[i + 1, :]
        for j in np.arange(i + 2, x.shape[0] - 1):
            a2 = x[j, :]
            b2 = x[j + 1, :]

            if _segments_intersect2d(a1, b1, a2, b2):
                intersections += 1
    return intersections


def spatial_similarity(emb, orig_pdist):
    # computes correlation between pairwise distances in embedding space and
    # pairwise distances in original space
    emb_pdist = pdist(emb, 'euclidean')
    return pearsonr(emb_pdist, orig_pdist)[0]


def optimize_embedding(video, immediate_recall, delayed_recall):
    save_dir = os.path.join(DATA_DIR, 'preprocessed', 'embeddings', 'opt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    avg_immediate = get_average_recall_events(video, immediate_recall)
    avg_delayed = get_average_recall_events(video, delayed_recall)

    n_neighbors = [10, 20, 50, 100]
    min_dist = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1]
    spread = [1, 2, 3, 4, 5]
    seed = np.arange(25)

    min_intersections = np.inf
    max_spatial_similarity = -np.inf
    best_params = {}

    video_pdist = pdist(video, metric='correlation')

    for nn in tqdm(n_neighbors, desc='n_neighbors'):
        for md in tqdm(min_dist, desc='min_dist'):
            for s in tqdm(spread, desc='spread'):
                for i in tqdm(seed, desc='seed'):
                    params = {
                        'metric': 'correlation',
                        'random_state': i,
                        'n_neighbors': nn,
                        'min_dist': md,
                        'spread': s,
                        'low_memory': False
                    }

                    fname = os.path.join(save_dir, f'{nn}_{md}_{s}_{i}.pkl')
                    if os.path.exists(fname):
                        with open(fname, 'rb') as f:
                            next_intersections, next_spatial_similarity = pickle.load(f)

                    else:
                        warnings.simplefilter('ignore')
                        np.random.seed(i)
                        embeddings = hyp.reduce([video, avg_immediate, avg_delayed, *immediate_recall, *delayed_recall],
                                                reduce={'model': 'UMAP', 'params': params}, ndims=2)

                        next_intersections = n_intersections(embeddings[0])
                        next_spatial_similarity = spatial_similarity(embeddings[0], video_pdist)

                        with open(fname, 'wb') as f:
                            pickle.dump([next_intersections, next_spatial_similarity], f)

                    if next_intersections == min_intersections:
                        if next_spatial_similarity > max_spatial_similarity:
                            max_spatial_similarity = next_spatial_similarity
                            best_params = params.copy()
                            print(f'\nnew optimum found [improved spatial match]: {min_intersections} intersections: {best_params}')

                    if next_intersections < min_intersections:
                        min_intersections = next_intersections
                        best_params = params.copy()
                        print(f'\nnew optimum found [fewer intersections]: {min_intersections} intersections: {best_params}')

    print(f'best params ({min_intersections} intersections): {best_params}')
    return best_params


def test_consistency(recall_embeddings, step=None):
    if step is None:
        step = STEP

    # create 2D grid
    X, Y = np.meshgrid(np.arange(XGRID_SCALE[0], XGRID_SCALE[1], step),
                       np.arange(YGRID_SCALE[0], YGRID_SCALE[1], step))

    # turn trajectory into a list of line segments
    seglist = []
    for sub in recall_embeddings:
        for i in range(sub.shape[0] - 1):
            p1 = Point(coord=sub[i, :])
            p2 = Point(coord=sub[i + 1, :])
            seg = LineSegment(p1=p1, p2=p2)
            seglist.append(seg)

    # compute the average vector and p-value at each grid point
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    P = np.zeros_like(X)
    C = np.zeros_like(X)

    pbar = tqdm(total=len(X) ** 2, leave=False)
    # noinspection PyShadowingNames
    for i, (x, y) in enumerate(zip(X, Y)):
        for j, (xi, yi) in enumerate(zip(x, y)):
            U[i, j], V[i, j], P[i, j], C[i, j] = compute_coord(xi, yi,
                                                               step * 2,
                                                               seglist,
                                                               kind='circle')
            pbar.update()
    pbar.close()

    # correct for multiple comparisons
    THRESH = .05
    Pc = multipletests(P.ravel(), method='fdr_bh', alpha=THRESH)[1].reshape(P.shape)

    M = np.hypot(U, V)
    M = plt.cm.Blues(M)
    # not significant after multiple comparisons correction; color grey
    M[Pc >= THRESH] = [.5, .5, .5, .25]
    # no trajectories pass through here; color transparent
    M[P == 1] = [.5, .5, .5, 0]

    return X, Y, U, V, M


def plotter(average, cmap=None, recalls=None, title='', step=None):
    if cmap is None:
        cmap = plt.cm.Spectral

    zorder = 1
    fig = plt.figure()
    ax = plt.gca()

    # add quiver plot (hypothesis testing)
    if recalls is not None:
        X, Y, U, V, M = test_consistency(recalls, step=step)
        ax.quiver(X, Y, U, V,
                    color=M.reshape(M.shape[0] * M.shape[1], 4),
                    zorder=zorder,
                    width=0.004)
        zorder += 1

    # plot trajectory coordinates
    ax.plot(average[:, 0], average[:, 1], zorder=zorder, c='k', alpha=0.5)
    zorder += 1

    # plot event markers
    ax.scatter(average[:, 0], average[:, 1], zorder=zorder, c='k', cmap=cmap, s=200)
    ax.scatter(average[:, 0], average[:, 1], zorder=zorder+1, c=range(average.shape[0]), cmap=cmap, s=150)

    # formatting stuff
    ax.axis('off')
    ax.set_title(title)
    ax.set_xlim(XGRID_SCALE)
    ax.set_ylim(YGRID_SCALE)

    return fig


# noinspection PyShadowingNames
def plot_trajectories(video, immediate_recall, delayed_recall, cmap=None):
    save_dir = os.path.join(DATA_DIR, 'preprocessed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fname = os.path.join(save_dir, 'reduced_embeddings.pkl')

    immediate_recall = [i for i in immediate_recall if np.prod(np.shape(i)) > 0]
    delayed_recall = [d for d in delayed_recall if np.prod(np.shape(d)) > 0]

    if not os.path.exists(fname):
        best_params = optimize_embedding(video, immediate_recall, delayed_recall)
        avg_immediate = get_average_recall_events(video, immediate_recall)
        avg_delayed = get_average_recall_events(video, delayed_recall)

        seed = best_params['random_state']
        np.random.seed(seed)
        embeddings = hyp.reduce([video, avg_immediate, avg_delayed, *immediate_recall, *delayed_recall],
                                reduce={'model': 'UMAP', 'params': best_params}, ndims=2)

        with open(fname, 'wb') as f:
            pickle.dump(embeddings, f)

    with open(fname, 'rb') as f:
        embeddings = pickle.load(f)

    video_embedding = embeddings[0]
    avg_immediate = embeddings[1]
    avg_delayed = embeddings[2]
    immediate = embeddings[3:(3 + len(immediate_recall))]
    delayed = embeddings[(3 + len(immediate_recall)):]

    # plot video events
    f1 = plotter(video_embedding, cmap=cmap, title='Video events')

    # plot immediate recalls
    f2 = plotter(avg_immediate, cmap=cmap, recalls=immediate, title='Immediate recall')

    # plot delayed recalls
    f3 = plotter(avg_delayed, cmap=cmap, recalls=delayed, title='Delayed recall')

    return f1, f2, f3

# # train model on sliding windows from video
# video_transcript = dw.io.load(os.path.join(DATA_DIR, 'task', 'storytext.txt'), dtype='text').lower()
#
# np.random.seed(1)
# n_topics = 50
# width = 50
# lda_params = {'model': 'LatentDirichletAllocation', 'args': [], 'kwargs': {'n_components': n_topics}}
# x, lda = dw.zoo.text.apply_text_model(['CountVectorizer', lda_params], sliding_windows(video_transcript, width=width,
#                                                                                        end=None),
#                                       return_model=True)
#
# data = get_formatted_data()
# _, _, _, transcript_events, immediate_events, delayed_events = \
#         get_video_and_recall_trajectories(data['experiment'], width=width, end=None, doc_model=lda, doc_name='LDA',
#                                           window_model=lda, window_name='LDA')
#
# f1, f2, f3 = plot_trajectories(transcript_events, immediate_events, delayed_events)
#
# f1.savefig('video.pdf')
# f2.savefig('immediate.pdf')
# f3.savefig('delayed.pdf')
