# noinspection PyPackageRequirements
import datawrangler as dw
import os
import sys
import numpy as np
import pandas as pd
import ast
import json
import datetime
import quail
import nltk
import warnings
import pickle
import datetime as dt

# noinspection PyPackageRequirements
from spellchecker import SpellChecker
from glob import glob as lsdir
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance, pearsonr, zscore
from sklearn.linear_model import LinearRegression
from flair.models import TextClassifier
from flair.data import Sentence

import brainiak.eventseg.event as event

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

nltk.download('omw-1.4')

def load_raw():
    datadir = os.path.join(DATA_DIR, 'raw_formatted')
    files = lsdir(os.path.join(datadir, '*.csv'))
    skip = ['data_descriptors.csv', 'event_descriptors.csv', 'id_filename_key.csv']
    # noinspection PyShadowingNames
    files = [f for f in files if os.path.split(f)[-1] not in skip]

    # noinspection PyShadowingNames
    raw_data = [pd.read_csv(f) for f in files]

    loaded = []
    subjects = []
    for i, x in enumerate(raw_data):
        # noinspection PyBroadException
        try:
            y = x.pivot_table(index=['datetime'], columns='variable', aggfunc=lambda a: a)
            y.columns = [c[1] for c in y.columns]
            loaded.append(y)
            subjects.append(f'P{i}')
        except:
            print(f'error loading data: {files[i]}')
            pass
    return loaded, subjects


def parse_data(d):
    datadir = os.path.join(DATA_DIR, 'raw_formatted')
    non_exp_descriptors = pd.read_csv(os.path.join(datadir, 'data_descriptors.csv'))
    exp_descriptors = pd.read_csv(os.path.join(datadir, 'event_descriptors.csv'))

    def variable_type(v):
        def helper(descriptors):
            if 'variable name' in descriptors.columns:
                field = 'variable name'
            else:
                field = 'exp_event'
            # noinspection PyShadowingNames
            inds = np.where([x.strip() == v.strip() for x in descriptors[field].values])[0]
            if len(inds) > 0:
                description = descriptors.iloc[inds[0]]['description']
            else:
                raise Exception('description not found')

            if any([keyword in description for keyword in ['fitbit', 'sleep', 'activ', 'sedent', 'elevation',
                                                           'floors', 'step', 'battery', 'cardio']]):
                return 'fitbit'
            elif any([keyword in v.lower() for keyword in ['fb_', 'cal', 'bodyfat', 'water', 'peak', 'weight', 'hr',
                                                           'oor', 'sync', 'device', 'bmi']]):
                return 'fitbit'
            elif any([keyword in description for keyword in ['clear', 'instruction', 'difficult', 'language', 'gender',
                                                             'coffee', 'color', 'today', 'plan', 'motiv', 'year',
                                                             'current', 'degree', 'freq', 'feedback', 'race',
                                                             'stress', 'impair']]):
                return 'survey'
            elif any([keyword in v.lower() for keyword in ['freq', 'setting']]):
                return 'survey'
            elif any([((keyword in description) or (keyword in v.lower())) for keyword in ['pres', 'rec', 'task',
                                                                                           'word', 'position', 'delay',
                                                                                           'movie', 'experiment']]):
                return 'experiment'
            else:
                return 'meta'

        if v.lower() == 'utc':
            return 'meta'
        elif v.lower() == 'tracker_features':
            return 'fitbit'
        elif v.lower() in ['recent_meds_injuries', 'job_activity', 'tracker_sync_today', 'typical_stress']:
            return 'survey'
        elif v.lower() in ['movie_sent_recall', 'movie_sent_recall_delay']:
            return 'experiment'

        # noinspection PyBroadException
        try:
            return helper(non_exp_descriptors)
        except:
            # noinspection PyBroadException
            try:
                return helper(exp_descriptors)
            except:
                if any([keyword in v.lower() for keyword in ['pres', 'rec', 'task', 'word', 'position', 'delay',
                                                             'resp']]):
                    return 'experiment'
                else:
                    return 'untagged'

    parsed = {}
    for c in d.columns:
        x = variable_type(c)
        if x in parsed.keys():
            parsed[x] = parsed[x].merge(pd.DataFrame(d[c]), how='outer', right_index=True, left_index=True)
        else:
            parsed[x] = pd.DataFrame(d[c])
    return parsed


def simplify_dict_list(x, subjs):
    combined = {'participants': subjs}
    for i, d in enumerate(x):
        for k in d.keys():
            if k in combined.keys():
                combined[k].append(d[k])
            else:
                combined[k] = [d[k]]
    return combined


def get_stats(parsed, stat_dict):
    stacked = {}
    subjs = parsed.pop('participants', None)
    for k in parsed.keys():
        stacked[k] = dw.stack(parsed[k], keys=subjs)

    stats = pd.DataFrame(columns=list(stat_dict.keys()), index=subjs)
    for s in stat_dict.keys():
        stats[s] = stat_dict(s)(stacked)

    return stats


def compute_recent_and_change(x, index, name, ref_days, base_days, today=None):
    # noinspection PyShadowingNames
    def average(d, f):
        warnings.simplefilter('ignore')
        return pd.Series(index=index,
                         data=[np.nanmean([eval(j) for j in i[f] if type(j) is str])
                               if (type(i) is pd.DataFrame and i.shape[0] > 0 and f in i.columns)
                               else np.nan for i in d])

    results = {'recent': average(extract_days_prior(x, ref_days, today=today), name)}

    # noinspection PyShadowingNames, PyTypeChecker
    baseline = average(extract_days_prior(x, base_days, today=[t - dt.timedelta(days=ref_days) for t in today]), name)
    results['recent / baseline'] = results['recent'] / baseline

    return results


def dict_diff(a, b):
    keys = list(set(a.keys()).union(set(b.keys())))
    diffs = {}
    for k in keys:
        diffs[k] = a[k] - b[k]
    return diffs


def fitness_stats(parsed, reference_days=7, baseline_days=180):
    stats = {}
    index = np.arange(len(parsed['fitbit']))

    # static body stats

    # bmi
    stats['BMI'] = pd.Series(index=index,
                             data=[eval(x) if type(x) is str and not np.isclose(eval(x), 0.0) else np.nan for x in
                                   get_raw_feature(parsed['fitbit'], 'bmi')])

    # bodyfat
    stats['body fat'] = pd.Series(index=index,
                                  data=[eval(x) if type(x) is str and not np.isclose(eval(x), 0.0) else np.nan for x in
                                        get_raw_feature(parsed['fitbit'], 'bodyfat')])


    # weight
    stats['weight'] = pd.Series(index=index,
                                data=[eval(x) if type(x) is str and not np.isclose(eval(x), 0.0) else np.nan for x in
                                      get_raw_feature(parsed['fitbit'], 'weight')])


    # dynamic body stats (for each, compute most recent + change in reference vs. baseline)

    # resting heart rate
    stats['resting heart rate'] = compute_recent_and_change(parsed['fitbit'], index, 'resting_HR', reference_days,
                                                            baseline_days, today=get_test_day(parsed['experiment']))

    # sleep hours
    stats['sleep duration'] = compute_recent_and_change(parsed['fitbit'], index, 'sleep_duration', reference_days,
                                                        baseline_days, today=get_test_day(parsed['experiment']))

    # sleep efficiency
    stats['sleep efficiency'] = compute_recent_and_change(parsed['fitbit'], index, 'sleep_duration', reference_days,
                                                          baseline_days, today=get_test_day(parsed['experiment']))


    # activity summary (recent + change in reference vs. baseline)

    # steps
    stats['steps'] = compute_recent_and_change(parsed['fitbit'], index, 'steps', reference_days, baseline_days,
                                               today=get_test_day(parsed['experiment']))

    # distance
    stats['distance'] = compute_recent_and_change(parsed['fitbit'], index, 'distance', reference_days, baseline_days,
                                                  today=get_test_day(parsed['experiment']))

    # elevation
    stats['elevation'] = compute_recent_and_change(parsed['fitbit'], index, 'elevation', reference_days, baseline_days,
                                                   today=get_test_day(parsed['experiment']))

    # floors
    stats['floors climbed'] = compute_recent_and_change(parsed['fitbit'], index, 'floors', reference_days,
                                                        baseline_days, today=get_test_day(parsed['experiment']))

    # activity details (recent + change in reference vs. baseline)

    # light activity minutes
    stats['light activity'] = compute_recent_and_change(parsed['fitbit'], index, 'light_act_mins', reference_days,
                                                        baseline_days, today=get_test_day(parsed['experiment']))

    # fairly active minutes
    stats['fair activity'] = compute_recent_and_change(parsed['fitbit'], index, 'fair_act_mins', reference_days,
                                                       baseline_days, today=get_test_day(parsed['experiment']))

    # very active minutes
    stats['high intensity activity'] = compute_recent_and_change(parsed['fitbit'], index, 'very_act_mins',
                                                                 reference_days, baseline_days,
                                                                 today=get_test_day(parsed['experiment']))

    # cal - cal_bmr
    cal = compute_recent_and_change(parsed['fitbit'], index, 'cal', reference_days, baseline_days,
                                    today=get_test_day(parsed['experiment']))
    cal_bmr = compute_recent_and_change(parsed['fitbit'], index, 'cal_bmr', reference_days, baseline_days,
                                        today=get_test_day(parsed['experiment']))
    stats['excess calories'] = dict_diff(cal, cal_bmr)

    # heart-specific activity details (recent + change in reference vs. baseline)

    # out of range minutes
    stats['out-of-range HR'] = compute_recent_and_change(parsed['fitbit'], index, 'oor_mins', reference_days,
                                                         baseline_days, today=get_test_day(parsed['experiment']))

    # fat burn minutes
    stats['fat burn HR'] = compute_recent_and_change(parsed['fitbit'], index, 'fb_mins', reference_days, baseline_days,
                                                     today=get_test_day(parsed['experiment']))

    # cardio minutes
    stats['cardio HR'] = compute_recent_and_change(parsed['fitbit'], index, 'cardio_mins', reference_days,
                                                   baseline_days, today=get_test_day(parsed['experiment']))

    # peak minutes
    stats['peak HR'] = compute_recent_and_change(parsed['fitbit'], index, 'peak_mins', reference_days, baseline_days,
                                                 today=get_test_day(parsed['experiment']))

    # today's heart rate variability (average) -- cannot compute change
    test_day = extract_days_prior(parsed['fitbit'], 1, today=get_test_day(parsed['experiment']))
    hrv = pd.Series(index=index)
    for i, x in enumerate(test_day):
        if 'todayHRval' in x.columns:
            hrv.loc[index[i]] = np.nanstd([eval(h) if type(h) is str else np.nan for h in x['todayHRval']])

    stats['HR variability'] = hrv

    # not including the following-- almost no one logged them:
    #   - food and water intake (recent + change in reference vs. baseline)
    #   - water logged
    #   - food calories logged
    x = alt_dict2df(stats)
    return pd.DataFrame(index=parsed['participants'], data=x.values, columns=x.columns)


def lemmatize(word, lemmatizer=None):
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    if type(word) == list:
        return [lemmatize(w, lemmatizer=lemmatizer) for w in word]

    tag = nltk.pos_tag([word])[0][1]
    if tag == 'J':
        pos = wordnet.ADJ
    elif tag == 'V':
        pos = wordnet.VERB
    elif tag == 'R':
        pos = wordnet.ADV
    else:
        pos = wordnet.NOUN

    return lemmatizer.lemmatize(word, pos)


# noinspection PyShadowingNames
def get_list_items(data, lists=None, pres_prefix='', rec_prefix='', aggregate_presentations=False, debug=False):
    if lists is None:
        lists = [1, 2, 3, 4]

    spell = SpellChecker(language='en')

    wordpool = pd.read_csv(os.path.join(DATA_DIR, 'task', 'wordpool.csv'))
    known_mistakes = pd.read_csv(os.path.join(DATA_DIR, 'task', 'spellcheck.csv'))

    def get_features(word):
        if ', ' in word:
            return [get_features(w) for w in word.split(', ')]

        # remove extraneous characters
        extras = [',', '.', '!', '?', ' ']
        word = ''.join([c for c in word if c not in extras])

        # basic spelling correction
        if type(word) is str:
            word = spell.correction(word.capitalize())
        else:
            raise ValueError(f'cannot process words of type {type(word)}')

        # known mistakes
        mistake = known_mistakes.query(f'misspelled == "{word.upper()}"')
        if len(mistake) > 0:
            word = mistake['corrected'].values[0]

        w = wordpool.query(f'WORD == "{word.upper()}"')
        if len(w) == 0:
            # try lemmatizing the word
            lemmatized_word = lemmatize(word.lower())
            lw = wordpool.query(f'WORD == "{lemmatized_word.upper()}"')
            if len(lw) > 0:
                w = lw
                word = lemmatized_word

        if len(w) == 0:
            if debug:
                print(f'unrecognized word: {word.upper()}')
            return {'item': word.upper(),
                    'word_length': len(word),
                    'starting_letter': word[0].upper()}
        else:
            return {'item': word.upper(),
                    'word_length': len(word),
                    'starting_letter': word[0].upper(),
                    'category': w['CATEGORY'].values[0].upper(),
                    'size': w['SIZE'].values[0].upper()}

    pres_words = []
    rec_words = []
    for subj_data in data:
        list_presentations = []
        list_recalls = []

        # noinspection PyBroadException
        try:
            for i, x in enumerate(lists):
                presented_items = [get_features(w) for w in subj_data[f'{pres_prefix}{x}'] if type(w) is not float]
                if aggregate_presentations:
                    list_presentations.extend([dw.core.update_dict(i, {'list': x}) for i in presented_items])
                    if i == 0:
                        try:
                            list_recalls.extend([get_features(w) for w in subj_data[f'{rec_prefix}'] if type(w) is not float])
                        except KeyError:
                            list_recalls.extend([])
                else:
                    list_presentations.append(presented_items)
                    try:
                        next_recalls = []
                        for w in subj_data[f'{rec_prefix}{x}']:
                            if type(w) is str:
                                next_features = get_features(w)
                                if type(next_features) is dict:
                                    next_recalls.append(next_features)
                                elif type(next_features) is list:
                                    next_recalls.extend(next_features)
                        list_recalls.append(next_recalls)
                    except KeyError:
                        list_recalls.append([])

            if aggregate_presentations:
                pres_words.append([list_presentations])
                rec_words.append([list_recalls])
            else:
                pres_words.append(list_presentations)
                rec_words.append(list_recalls)
        except:
            raise Exception('throwing this error to help with debugging...')
    return quail.Egg(pres=pres_words, rec=rec_words)


def sliding_windows(text, width=10, end='.'):
    punctuation = ['.', ',', '-', '?', '!']

    if len(text) == 0:
        return None
    elif type(text) is list:
        return [sliding_windows(t, width=width, end=end) for t in text]

    windows = []

    if (end is None) or (len(end) == 0):
        parts = text.split()
        end = ''
    else:
        parts = text.split(end)

    for p in punctuation:
        windows = [w.replace(p, '') for w in windows]

    for i in range(np.max([len(parts) - width, 1])):
        windows.append((end + ' ').join([p.strip() for p in parts[i:np.min([i + width, len(parts)])]]) + end)

    windows = [w.strip().lower() for w in windows if len(w.strip()) > 1]

    return [w for w in windows if len(w) > 0]



# noinspection PyTypeChecker
def get_events(transcript, model, width=10, end='.', max_k=50):
    # source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/sherlock_helpers/
    # sherlock_helpers/functions.py
    def create_diag_mask(arr, diag_start=0, diag_limit=None):
        diag_mask = np.zeros_like(arr, dtype=bool)
        if diag_limit is None:
            diag_limit = find_diag_limit(arr)

        # noinspection PyShadowingNames
        for k in range(diag_start, diag_limit):
            ix = kth_diag_indices(diag_mask, k)
            diag_mask[ix] = True

        return diag_mask

    # source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/sherlock_helpers/
    # sherlock_helpers/functions.py
    # noinspection PyShadowingNames
    def find_diag_limit(arr):
        for k in range(arr.shape[0]):
            d = np.diag(arr, k=k)
            if ~(d > 0).any():
                return k

    # source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/sherlock_helpers/
    # sherlock_helpers/functions.py
    # noinspection PyShadowingNames
    def kth_diag_indices(arr, k):
        row_ix, col_ix = np.diag_indices_from(arr)
        if k == 0:
            return row_ix, col_ix
        else:
            return row_ix[:-k], col_ix[k:]

    # source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/notebooks/main/
    # eventseg_analysis.ipynb
    # noinspection PyShadowingNames
    def reduce_model(m, ev):
        """
        Reduce a model based on event labels
        """
        w = (np.round(ev.segments_[0]) == 1).astype(bool)
        return np.array([m[wi, :].mean(axis=0) for wi in w.T])

    if type(transcript) is list:
        return [get_events(t, model, width=width, end=end) for t in transcript]

    embeddings = dw.wrangle(sliding_windows(transcript, width=width, end=end), text_kwargs={'model': model}).values
    ks = list(range(2, np.min([embeddings.shape[0], max_k])))

    # source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/notebooks/main/
    # eventseg_analysis.ipynb
    mcorr = np.corrcoef(embeddings)
    scores = []
    for k in ks:
        ev = event.EventSegment(k)
        ev.fit(embeddings)
        i1, i2 = np.where(np.round(ev.segments_[0]) == 1)
        w = np.zeros_like(ev.segments_[0])
        w[i1, i2] = 1
        mask = np.dot(w, w.T).astype(bool)

        # Create mask such that the maximum temporal distance
        # for within and across correlations is the same
        local_mask = create_diag_mask(mask)

        within_vals = np.reshape(mcorr[mask * local_mask], [-1, 1])
        across_vals = np.reshape(mcorr[~mask * local_mask], [-1, 1])
        try:
            scores.append(wasserstein_distance(within_vals.ravel(), across_vals.ravel()))
        except ValueError:
            scores.append(-np.inf)


    try:
        if np.all(np.isinf(scores)):
            raise ValueError('cannot segment events')
        opt_k = ks[np.argmax(scores)]
        ev = event.EventSegment(opt_k)
        ev.fit(embeddings)
        return reduce_model(embeddings, ev)
    except ValueError:
        # noinspection PyBroadException
        try:
            return np.atleast_2d(embeddings.mean(axis=0))
        except:
            return np.array([])


# source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/sherlock_helpers/sherlock_helpers/
# functions.py
def r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))


# source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/sherlock_helpers/sherlock_helpers/
# functions.py
def z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


# source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/sherlock_helpers/sherlock_helpers/
# functions.py
def corr_mean(rs, axis=0):
    return z2r(np.nanmean([r2z(r) for r in rs], axis=axis))


# source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/notebooks/main/
# precision_distinctiveness_fig.ipynb
def precision(video, recall):
    if type(recall) is list:
        return pd.Series(index=np.arange(len(recall)), data=[precision(video, r) for r in recall])
    if np.prod(recall.shape) == 0:
        return np.nan
    return corr_mean(np.max(1 - cdist(video, recall, 'correlation'), 0))


# source: https://github.com/ContextLab/sherlock-topic-model-paper/blob/master/code/notebooks/main/
# precision_distinctiveness_fig.ipynb
# noinspection PyArgumentList
def distinctiveness(video, recall):
    if type(recall) is list:
        return pd.Series(index=np.arange(len(recall)), data=[distinctiveness(video, r) for r in recall])

    if np.prod(recall.shape) == 0:
        return np.nan

    corrmat = 1 - cdist(video, recall, 'correlation')
    z_corrs = zscore(corrmat, axis=0)
    return z_corrs.max(axis=0).mean()


def get_pres_inds(items, presented_items, inds, exclude_nans=True):
    pres_inds = []
    for i in inds:
        next_matches = [j for j, w in enumerate(items) if w.upper() == presented_items[i].upper()]
        if (not exclude_nans) and len(next_matches) == 0:
            pres_inds.append(np.nan)
        elif len(next_matches) == 1:
            pres_inds.extend(next_matches)
        else:
            pres_inds.extend(next_matches)
    return np.array(pres_inds)


def get_temporal_clustering_vocab(correct, items, presented_items):
    adjacent = []
    for i in range(len(presented_items) - 1):
        adjacent.append([get_pres_inds(items, presented_items, [i, i+1], exclude_nans=False)])

    n_both_correct = 0
    correction = 0
    for a in adjacent:
        try:
            if correct[int(a[0][0])] and correct[int(a[0][1])]:
                n_both_correct += 1
        except (TypeError, ValueError):
            correction += 1

    if np.sum(correct) > (correction + 1):
        return n_both_correct / (np.sum(correct) - 1 - correction)
    else:
        return np.nan


def get_mean_error_dist(responses, presented_items):
    correct_positions = get_pres_inds([r[3] for r in responses], presented_items, range(len(presented_items)))
    observed_positions = get_pres_inds([r[0] for r in responses], presented_items, range(len(presented_items)))

    diffs = np.abs(correct_positions - observed_positions)
    return np.mean(diffs[diffs > 0])


# noinspection PyShadowingNames
def spatial_estimation_error(data, n, metric='mean'):
    if type(n) is list:
        if len(n) == 0:
            return np.nan

        errors = spatial_estimation_error(data, n[0], metric=metric)
        for i in n[1:]:
            errors = errors + spatial_estimation_error(data, i, metric=metric)
        return errors / len(n)

    errors = pd.Series(index=np.arange(len(data)))
    for s in range(len(data)):
        pres = [eval(x) for x in data[s][f'spatial_pres_{n}'] if type(x) is str]
        resp = [eval(x) for x in data[s][f'spatial_resp_{n}'] if type(x) is str]

        next_errors = []
        for i in range(len(pres)):
            target_positions = {x[2]: [x[0], x[1]] for x in pres[i]}
            response_positions = {x[2]: [x[0], x[1]] for x in resp[i][-1]}

            trial_errors = []
            for k in target_positions.keys():
                trial_errors.append(float(cdist(np.atleast_2d(target_positions[k]),
                                                np.atleast_2d(response_positions[k]))))
            next_errors.append(np.mean(trial_errors))

        if metric == 'mean':
            errors[s] = np.mean(next_errors)
        elif metric == 'var':
            errors[s] = np.var(next_errors)
        elif metric == 'std':
            errors[s] = np.std(next_errors)
        else:
            raise ValueError(f'unknown metric: {metric}')
    return errors


def dict2df(d):
    # source: https://stackoverflow.com/questions/24988131/
    # nested-dictionary-to-multiindex-dataframe-where-dictionary-keys-are-column-label
    def flatten_dict(dictionary, t=tuple(), keys=None):
        if keys is None:
            keys = {}

        for key, val in dictionary.items():
            t = t + (key,)
            if isinstance(val, dict):
                flatten_dict(val, t, keys)
            else:
                keys.update({t: val})
            t = t[:-1]
        return keys

    return pd.DataFrame(flatten_dict(d))


def alt_dict2df(stats):
    # convert stats to a dataframe
    keys = list(stats.keys())
    assert type(stats[keys[0]]) is pd.Series, 'first key must be a series; cannot flatten stats dictionary'

    df = pd.DataFrame({keys[0]: stats[keys[0]].values}, index=stats[keys[0]].index)
    merged_columns = [('', keys[0])]
    for k in keys[1:]:
        if type(stats[k]) is pd.Series:
            merged_columns.append(('', k))
            next_df = pd.DataFrame({k: stats[k].values}, index=stats[k].index)
        elif type(stats[k]) is pd.DataFrame:
            columns = [(k, c) for c in stats[k].columns]
            merged_columns.extend(columns)
            next_df = pd.DataFrame(stats[k].values, columns=columns)
        elif type(stats[k]) is dict:
            s = dict2df(stats[k])
            columns = [(k, c[0]) for c in s.columns]
            merged_columns.extend(columns)
            next_df = pd.DataFrame(s.values, columns=columns)
        else:
            raise ValueError(f'unsupported datatype ({k}): {type(stats[k])}')
        df = df.merge(next_df, how='left', left_index=True, right_index=True)
    return pd.DataFrame(df.values, columns=pd.MultiIndex.from_tuples(merged_columns))


# noinspection PyTypeChecker
def get_video_and_recall_trajectories(data, width=10, end='.', doc_model=None, doc_name=None, window_model=None, window_name=None):
    if doc_model is None:
        doc_model = {'model': 'TransformerDocumentEmbeddings', 'args': ['bert-base-uncased'], 'kwargs': {}}
    if doc_name is None:
        doc_name = doc_model['args'][0]

    if window_model is None:
        window_model = {'model': 'SentenceTransformerDocumentEmbeddings', 'args': ['stsb-bert-large'], 'kwargs': {}}
    if window_name is None:
        window_name = window_model['args'][0]

    preprocessing_dir = os.path.join(DATA_DIR, 'preprocessed')

    transcript = dw.io.load(os.path.join(DATA_DIR, 'task', 'storytext.txt'), dtype='text').lower()
    immediate_recall = []
    delayed_recall = []
    for s in range(len(data)):
        if 'movie_sent_recall' in data[s].keys():
            immediate_recall.append([r.strip().lower() for r in data[s]['movie_sent_recall']
                                     if type(r) is str and len(r) > 1])
        else:
            immediate_recall.append([])

        if 'movie_sent_recall_delay' in data[s].keys():
            delayed_recall.append([r.strip().lower() for r in data[s]['movie_sent_recall_delay']
                                   if type(r) is str])
        else:
            delayed_recall.append([])

    immediate_transcripts = [' '.join(x) for x in immediate_recall]
    delayed_transcripts = [' '.join(x) for x in delayed_recall]

    embedding_dir = os.path.join(preprocessing_dir, 'embeddings')
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    embeddings_fname = os.path.join(embedding_dir, f'embeddings_{doc_name}.pkl')
    if not os.path.exists(embeddings_fname):
        transcript_embedding = dw.wrangle(sliding_windows(transcript, width=width, end=end), text_kwargs={'model': doc_model})
        immediate_embeddings = dw.wrangle(sliding_windows(immediate_transcripts, width=width, end=end),
                                          text_kwargs={'model': doc_model})
        delayed_embeddings = dw.wrangle(sliding_windows(delayed_transcripts, width=width, end=end),
                                        text_kwargs={'model': doc_model})

        with open(embeddings_fname, 'wb') as f:
            pickle.dump([transcript_embedding, immediate_embeddings, delayed_embeddings], f)

    with open(embeddings_fname, 'rb') as f:
        transcript_embedding, immediate_embeddings, delayed_embeddings = pickle.load(f)

    trajectories_fname = os.path.join(embedding_dir, f'trajectories_{window_name}.pkl')

    if not os.path.exists(trajectories_fname):
        transcript_events = get_events(transcript, window_model, width=width, end=end)
        immediate_events = get_events(immediate_transcripts, window_model, width=width, end=end)
        delayed_events = get_events(delayed_transcripts, window_model, width=width, end=end)

        with open(trajectories_fname, 'wb') as f:
            pickle.dump([transcript_events, immediate_events, delayed_events], f)

    with open(trajectories_fname, 'rb') as f:
        transcript_events, immediate_events, delayed_events = pickle.load(f)

    return transcript_embedding, immediate_embeddings, delayed_embeddings,\
           transcript_events, immediate_events, delayed_events


# noinspection PyTypeChecker
def behavioral_stats(parsed):
    warnings.simplefilter('ignore')
    stats = {task: {'immediate': {}, 'delayed': {}} for task in ['free recall', 'naturalistic recall',
                                                                 'vocab learning', 'spatial learning']}

    # free recall (immediate + delayed)
    immediate_fr = get_list_items(parsed['experiment'], lists=[1, 2, 3, 4], pres_prefix='pres_word_',
                                  rec_prefix='rec_word_')
    delayed_fr = get_list_items(parsed['experiment'], pres_prefix='pres_word_', rec_prefix='rec_word_delay',
                                aggregate_presentations=True)

    immediate_spc = immediate_fr.analyze('spc').data.groupby('Subject').mean()
    delayed_spc = delayed_fr.analyze('spc').data.groupby('Subject').mean()

    immediate_fingerprints = immediate_fr.analyze('fingerprint').data.groupby('Subject').mean()
    delayed_fingerprints = delayed_fr.analyze('fingerprint').data.groupby('Subject').mean()

    # proportion of words recalled
    stats['free recall']['immediate']['recall proportion'] = immediate_spc.mean(axis=1)
    stats['free recall']['delayed']['recall proportion'] = delayed_spc.mean(axis=1)

    # average primacy effect
    stats['free recall']['immediate']['primacy'] = \
        immediate_spc.iloc[:, :3].mean(axis=1) / immediate_spc.iloc[:, 5:10].mean(axis=1)
    stats['free recall']['delayed']['primacy'] = \
        delayed_spc.iloc[:, :16].mean(axis=1) / delayed_spc.iloc[:, 16:48].mean(axis=1)

    # average recency effect
    stats['free recall']['immediate']['recency'] = \
        immediate_spc.iloc[:, -3:].mean(axis=1) / immediate_spc.iloc[:, 5:10].mean(axis=1)
    stats['free recall']['delayed']['recency'] = \
        delayed_spc.iloc[:, -16:].mean(axis=1) / delayed_spc.iloc[:, 16:48].mean(axis=1)

    # average temporal clustering score
    stats['free recall']['immediate']['clustering: temporal'] = immediate_fingerprints['temporal']
    stats['free recall']['delayed']['clustering: temporal'] = delayed_fingerprints['temporal']
    stats['free recall']['delayed']['clustering: list'] = delayed_fingerprints['list']

    # average category clustering score
    stats['free recall']['immediate']['clustering: category'] = immediate_fingerprints['category']
    stats['free recall']['delayed']['clustering: category'] = delayed_fingerprints['category']

    # average size clustering score
    stats['free recall']['immediate']['clustering: size'] = immediate_fingerprints['size']
    stats['free recall']['delayed']['clustering: size'] = delayed_fingerprints['size']

    # average starting letter clustering score
    stats['free recall']['immediate']['clustering: starting letter'] = immediate_fingerprints['starting_letter']
    stats['free recall']['delayed']['clustering: starting letter'] = delayed_fingerprints['starting_letter']

    # average word length clustering score
    stats['free recall']['immediate']['clustering: word length'] = immediate_fingerprints['word_length']
    stats['free recall']['delayed']['clustering: word length'] = delayed_fingerprints['word_length']

    # movie task (immediate + delayed)

    # proportion of multiple choice questions answered correctly (immediate only)
    correct_responses = ['library', 'custodian', 'reading', 'professor', '1940s', '2', 'waxed the tops of bookshelves',
                         'never', '3am', 'less than high school', 'quiet']
    stats['naturalistic recall']['immediate']['proportion correct'] =\
        pd.Series(index=np.arange(len(parsed['experiment'])))
    for s in range(len(parsed['experiment'])):
        next_responses = [s.strip().lower() for s in
                          [eval(q) for q in parsed['experiment'][s]['movie_qs'] if type(q) is str][0]]
        if next_responses[0] != 'no':
            pass
        n_correct = np.sum([correct == resp for correct, resp in zip(correct_responses, next_responses[1:])])

        stats['naturalistic recall']['immediate']['proportion correct'][s] = n_correct / len(correct_responses)

    # average semantic similarity between full transcript and full response
    transcript_embedding, immediate_embeddings, delayed_embeddings,\
    transcript_events, immediate_events, delayed_events = \
        get_video_and_recall_trajectories(parsed['experiment'])

    immediate_match = 1 - cdist(transcript_embedding, immediate_embeddings, metric='correlation')[0]
    delayed_match = 1 - cdist(transcript_embedding, delayed_embeddings, metric='correlation')[0]

    stats['naturalistic recall']['immediate']['semantic match'] = pd.Series(index=np.arange(len(parsed['experiment'])))
    stats['naturalistic recall']['delayed']['semantic match'] =  pd.Series(index=np.arange(len(parsed['experiment'])))

    i = 0
    j = 0
    for s in np.arange(len(immediate_transcripts)):
        if len(immediate_transcripts[s].strip()) > 0:
            stats['naturalistic recall']['immediate']['semantic match'][s] = immediate_match[i]
            i += 1
        if len(delayed_transcripts[s].strip()) > 0:
            stats['naturalistic recall']['delayed']['semantic match'][s] = delayed_match[j]
            j += 1

    # average precision
    stats['naturalistic recall']['immediate']['precision'] = precision(transcript_events, immediate_events)
    stats['naturalistic recall']['delayed']['precision'] = precision(transcript_events, delayed_events)

    # average distinctiveness
    stats['naturalistic recall']['immediate']['distinctiveness'] = distinctiveness(transcript_events, immediate_events)
    stats['naturalistic recall']['delayed']['distinctiveness'] = distinctiveness(transcript_events, delayed_events)

    # number of detected events in response
    stats['naturalistic recall']['immediate']['n events'] = pd.Series(index=np.arange(len(parsed['experiment'])),
                                                                      data=[e.shape[0] for e in immediate_events])
    stats['naturalistic recall']['delayed']['n events'] = pd.Series(index=np.arange(len(parsed['experiment'])),
                                                                      data=[e.shape[0] for e in delayed_events])

    # number of sentences in response
    stats['naturalistic recall']['immediate']['n sentences'] = \
        pd.Series(index=np.arange(len(parsed['experiment'])), data=[len(t.split('.')) for t in immediate_transcripts])
    stats['naturalistic recall']['delayed']['n sentences'] = \
        pd.Series(index=np.arange(len(parsed['experiment'])), data=[len(t.split('.')) for t in delayed_transcripts])

    # average number of sentences per event in response
    stats['naturalistic recall']['immediate']['event length'] = \
        stats['naturalistic recall']['immediate']['n sentences'] / stats['naturalistic recall']['immediate']['n events']
    stats['naturalistic recall']['delayed']['event length'] = \
        stats['naturalistic recall']['delayed']['n sentences'] / stats['naturalistic recall']['delayed']['n events']

    # vocab learning (immediate + delayed)
    stats['vocab learning']['immediate'] = {'p(correct): all': [], 'p(correct): early': [], 'p(correct): late': [],
                                            'temporal clustering': [], 'error distance': [], 'similarity: correct': [],
                                            'similarity: incorrect': []}
    stats['vocab learning']['delayed'] = {'p(correct): all': [], 'p(correct): early': [], 'p(correct): late': [],
                                          'reaction time': [], 'speed/accuracy': [], 'temporal clustering': [],
                                          'error distance': [], 'similarity: correct': [],
                                          'similarity: incorrect': []}

    for s in range(len(parsed['experiment'])):
        pres_eng = [eval(p)[2] for p in parsed['experiment'][s]['vocab_pres'] if type(p) is str]
        pres_gle = [eval(p)[0] for p in parsed['experiment'][s]['vocab_pres'] if type(p) is str]

        resp = [eval(p) for p in parsed['experiment'][s]['vocab_resp'] if type(p) is str]
        resp_delayed = [eval(p) for p in parsed['experiment'][s]['vocab_resp_delay'] if type(p) is str]

        correct = [r[0] == r[3] for r in resp]
        correct_delayed = [r[0] == r[3] for r in resp_delayed]

        # average proportion correct
        stats['vocab learning']['immediate']['p(correct): all'].append(np.mean(correct))
        stats['vocab learning']['delayed']['p(correct): all'].append(np.mean(correct_delayed))

        # average early proportion correct
        early_inds = get_pres_inds([r[3] for r in resp], pres_gle, [0, 1, 2])
        early_inds_delayed = get_pres_inds([r[3] for r in resp_delayed], pres_gle, [0, 1, 2])

        stats['vocab learning']['immediate']['p(correct): early'].append(np.mean(np.array(correct)[early_inds]))
        stats['vocab learning']['delayed']['p(correct): early'].append(np.mean(np.array(correct_delayed)[
            early_inds_delayed]))

        # average late proportion correct
        late_inds = get_pres_inds([r[3] for r in resp], pres_gle, [7, 8, 9])
        late_inds_delayed = get_pres_inds([r[3] for r in resp_delayed], pres_gle, [7, 8, 9])

        stats['vocab learning']['immediate']['p(correct): late'].append(np.mean(np.array(correct)[late_inds]))
        stats['vocab learning']['delayed']['p(correct): late'].append(np.mean(np.array(correct_delayed)[
            late_inds_delayed]))

        # average reaction time (delayed only) -- not logged for immediate    TODO: look further into this...
        stats['vocab learning']['delayed']['reaction time'].append(np.mean([r[5] for r in resp_delayed]))

        # p(correct) / average reaction time (delayed only)
        stats['vocab learning']['delayed']['speed/accuracy'].append(
            stats['vocab learning']['delayed']['p(correct): all'][-1] /\
            stats['vocab learning']['delayed']['reaction time'][-1])

        # temporal clustering: p(correct next | correct current)
        stats['vocab learning']['immediate']['temporal clustering'].append(
            get_temporal_clustering_vocab(correct, [r[3] for r in resp], pres_gle))
        stats['vocab learning']['delayed']['temporal clustering'].append(
            get_temporal_clustering_vocab(correct_delayed, [r[3] for r in resp_delayed], pres_gle))

        # average error distance (how far away on the study list are errors?)
        stats['vocab learning']['immediate']['error distance'].append(
            get_mean_error_dist(resp, pres_gle))
        stats['vocab learning']['delayed']['error distance'].append(
            get_mean_error_dist(resp_delayed, pres_gle))

        glove = {'model': 'WordEmbeddings', 'args': ['glove'], 'kwargs': {}}
        eng_embeddings = dw.wrangle([w.lower() for w in pres_eng], text_kwargs={'model': glove})

        # average pairwise semantic similarity of correct words (vs. all)
        correct_inds = get_pres_inds([r[3] for i, r in enumerate(resp) if correct[i]], pres_gle, np.arange(10))
        correct_inds_delayed = get_pres_inds([r[3] for i, r in enumerate(resp_delayed) if correct_delayed[i]], pres_gle,
                                             np.arange(10))
        incorrect_inds = get_pres_inds([r[3] for i, r in enumerate(resp) if not correct[i]], pres_gle, np.arange(10))
        incorrect_inds_delayed = get_pres_inds([r[3] for i, r in enumerate(resp_delayed) if not correct_delayed[i]],
                                               pres_gle, np.arange(10))

        average_similarity = corr_mean(1 - pdist(eng_embeddings, metric='correlation'))

        stats['vocab learning']['immediate']['similarity: correct'].append(
            corr_mean(1 - pdist(eng_embeddings.loc[correct_inds], metric='correlation')) / average_similarity)
        stats['vocab learning']['delayed']['similarity: correct'].append(
            corr_mean(1 - pdist(eng_embeddings.loc[correct_inds_delayed], metric='correlation')) / average_similarity)

        # semantic pairwise similarity of errors (vs. all)
        stats['vocab learning']['immediate']['similarity: incorrect'].append(
            corr_mean(1 - pdist(eng_embeddings.loc[incorrect_inds], metric='correlation')) / average_similarity)
        stats['vocab learning']['delayed']['similarity: incorrect'].append(
            corr_mean(1 - pdist(eng_embeddings.loc[incorrect_inds_delayed], metric='correlation')) / average_similarity)

    for i in ['immediate', 'delayed']:
        for j in stats['vocab learning'][i].keys():
            stats['vocab learning'][i][j] = pd.Series(index=np.arange(len(parsed['experiment'])),
                                                      data=stats['vocab learning'][i][j])

    # spatial task (immediate only-- no delayed task)
    stats['spatial learning']['immediate'] = {}

    # average estimation error (2 or 3 shapes)
    stats['spatial learning']['immediate']['estimation error (2/3)'] = spatial_estimation_error(parsed['experiment'],
                                                                                                [2, 3])
    # average estimation error (4 or 5 shapes)
    stats['spatial learning']['immediate']['estimation error (4/5)'] = spatial_estimation_error(parsed['experiment'],
                                                                                                [4, 5])
    # average estimation error (6 or 7 shapes)
    stats['spatial learning']['immediate']['estimation error (6/7)'] = spatial_estimation_error(parsed['experiment'],
                                                                                                [6, 7])
    # slope of estimation error vs. number of shapes
    errors = pd.concat([spatial_estimation_error(parsed['experiment'], [i]) for i in range(2, 8)], axis=1).values

    slopes = pd.Series(index=np.arange(len(parsed['experiment'])))
    for s in range(len(parsed['experiment'])):
        reg = LinearRegression().fit(np.atleast_2d(np.arange(2, 8)).T, np.atleast_2d(errors[s, :]).T)
        slopes[s] = float(reg.coef_)
    stats['spatial learning']['immediate']['error change by n shapes'] = slopes

    # error (6 or 7 shapes) - error (2 or 3 shapes)
    stats['spatial learning']['immediate']['estimation error (6/7) - (2/3)'] = spatial_estimation_error(
        parsed['experiment'], [6, 7]) - spatial_estimation_error(parsed['experiment'], [2, 3])

    # average error variability across different numbers of shapes
    stats['spatial learning']['immediate']['error std dev (2/3)'] = spatial_estimation_error(parsed['experiment'],
                                                                                             [2, 3], metric='std')
    # average estimation error (4 or 5 shapes)
    stats['spatial learning']['immediate']['error std dev (4/5)'] = spatial_estimation_error(parsed['experiment'],
                                                                                             [4, 5], metric='std')
    # average estimation error (6 or 7 shapes)
    stats['spatial learning']['immediate']['error std dev (6/7)'] = spatial_estimation_error(parsed['experiment'],
                                                                                             [6, 7], metric='std')

    # turn the dictionary into a dataframe (with MultiIndex columns)
    x = dict2df(stats)
    return pd.DataFrame(index=parsed['participants'], data=x.values, columns=x.columns)


def str2dt(s):
    dt_format = '%Y-%m-%d %H:%M:%S.%f'
    return dt.datetime.strptime(s, dt_format)


# noinspection PyShadowingNames
def get_raw_feature(x, name, return_idx=False, truncate=True):
    if type(x) is list:
        return [get_raw_feature(i, name, return_idx=return_idx) for i in x]

    if name not in x.columns:
        if return_idx:
            return str2dt(x.index[-1]), 'np.nan'
        else:
            return 'np.nan'

    if return_idx:
        f = [(str2dt(idx), val) for idx, val in x[name].items() if type(val) is str]
    else:
        f = [i for i in x[name] if type(i) is str]
    if len(f) > 0:
        if truncate:
            return f[0]
        else:
            return f
    else:
        return np.nan


def get_indicator_feature(x, name):
    participant_vals = [[x.lower() for x in eval(r)] for r in get_raw_feature(x, name)]
    vals = []
    for x in participant_vals:
        for y in x:
            vals.append(y)
    vals = np.unique(vals)
    df = pd.DataFrame(columns=vals)

    for v in vals:
        df[v] = np.array([v in p for p in participant_vals], dtype=int)
    return df


def extract_days_prior(x, n_days, today=None):
    if type(x) is list:
        if type(today) is not list:
            today = [today] * len(x)
        return [extract_days_prior(i, n_days, today=t) for i, t in zip(x, today)]

    all_dates = [str2dt(i) for i in x.index.values]

    if today is None:
        today = all_dates[-1]

    interval_start = today - dt.timedelta(days=n_days)
    return x.loc[[d >= interval_start for d in all_dates]]


def get_test_day(experiment):
    if type(experiment) is list:
        return [get_test_day(x) for x in experiment]

    return np.sort([str2dt(d) for d in experiment.dropna(how='all').index.values])[0]


def get_tracked_exercise(x, minimum_exercise_mins):
    light_activity = get_raw_feature(x, 'light_act_mins', truncate=False)
    light_activity = [eval(x) if type(x) is str else np.nan for x in light_activity]

    medium_activity = get_raw_feature(x, 'fair_act_mins', truncate=False)
    medium_activity = [eval(x) if type(x) is str else np.nan for x in medium_activity]

    high_activity = get_raw_feature(x, 'very_act_mins', truncate=False)
    high_activity = [eval(x) if type(x) is str else np.nan for x in high_activity]
    return [(light >= minimum_exercise_mins) or
            (medium >= minimum_exercise_mins) or
            (high >= minimum_exercise_mins) for light, medium, high in
            zip(light_activity, medium_activity, high_activity)]


def get_days_exercised(x, min_mins):
    if type(x) is list:
        y = [get_days_exercised(i, min_mins) for i in x]
        return [i[0] for i in y], [i[1] for i in y]

    all_dates = [str2dt(i) for i in x.index.values]
    duration = all_dates[-1] - all_dates[0]

    activity_columns = [c for c in x.columns if c in ['light_act_mins', 'fair_act_mins', 'very_act_mins']]
    exercise = pd.DataFrame(x[activity_columns], dtype=float).sum(axis=1)

    last_date = None
    day_counter = 0
    for idx, val in exercise.items():
        if type(val) is str:
            v = eval(val)
        else:
            v = val
        if v > min_mins:
            if (last_date is None) or ((str2dt(idx) - last_date) >= dt.timedelta(days=1)):
                day_counter += 1
                last_date = str2dt(idx)
    return day_counter, duration


def get_sentiment(x, sentiment_classifier):
    if type(x) is list:
        return [get_sentiment(i, sentiment_classifier) for i in x]

    s = Sentence(x)
    sentiment_classifier.predict(s)

    if len(s.labels) == 0:
        return np.nan
    else:
        if s.labels[0].value == 'POSITIVE':
            return s.labels[0]._score
        elif s.labels[0].value == 'NEGATIVE':
            return -s.labels[0]._score
        else:
            pass


def survey_stats(parsed, baseline_days=30):
    index = np.arange(len(parsed['survey']))
    stats = {}

    # age
    birthyear = get_raw_feature(parsed['survey'], 'birthyear', return_idx=True)
    stats['age'] = pd.Series(index=index,
                             data=[int(np.round(b[0].year - eval(b[1]))) if not np.isnan(eval(b[1])) else np.nan
                                   for b in birthyear])

    # gender
    stats['gender'] = pd.get_dummies([g.lower() for g in get_raw_feature(parsed['survey'], 'gender')])

    # race
    stats['race'] = get_indicator_feature(parsed['survey'], 'race')

    # degree
    stats['degree'] = pd.get_dummies([x.lower() for x in get_raw_feature(parsed['survey'], 'degree')])

    # number of fluent languages
    stats['number fluent languages'] = pd.Series(index=index,
                                                 data=[len(eval(x)) for x in get_raw_feature(parsed['survey'],
                                                                                             'fluent_langs')])

    # number of familiar languages
    stats['number familiar languages'] = pd.Series(index=index,
                                                   data=[len(eval(x)) for x in get_raw_feature(parsed['survey'],
                                                                                           'familiar_langs')])

    # color vision
    stats['color vision'] = pd.Series(index=index,
                                      data=[1 if c == 'Yes' else 0 for c in get_raw_feature(parsed['survey'],
                                                                                            'color_vision')])

    # uncorrected visual impairments
    stats['vision impaired'] = pd.Series(index=index,
                                         data=[1 if c == 'Yes' else 0 for c in get_raw_feature(parsed['survey'],
                                                                                           'uncorr_impair')])

    # number of medications or injuries
    health_dict = {
        'anxiety or depression': ['zoloft', 'welbutrin', 'trazodone', 'xanax', 'zoloft', 'sertraline'],
        'high blood pressure': ['blood pressure', 'diuretic'],
        'bipolar': ['lamictal'],
        'hypothyroid': ['levo'],
        'unspecified medications': ['some medication'],
        'recent head injury': ['skull', 'concussion']}

    meds = [x.strip().lower() for x in get_raw_feature(parsed['survey'], 'recent_meds_injuries')]
    stats['health and wellness'] = pd.DataFrame(index=index, columns=list(health_dict.keys()))
    for i, m in enumerate(meds):
        for k in health_dict.keys():
            if any([x in m for x in health_dict[k]]):
                stats['health and wellness'].loc[i, k] = 1
            else:
                stats['health and wellness'].loc[i, k] = 0

    # self-reported behaviors and mental state

    # current stress level
    stress_dict = {'very relaxed': -2,
                   'a little relaxed': -1,
                   'neutral': 0,
                   'a little stressed': 1,
                   'very stressed': 2}
    stats['current stress'] = pd.Series(index=index,
                                        data=[stress_dict[k.lower()] for k in get_raw_feature(parsed['survey'],
                                                                                              'current_stress')])

    # typical stress level
    stats['typical stress'] = pd.Series(index=index,
                                        data=[stress_dict[k.lower()] for k in get_raw_feature(parsed['survey'],
                                                                                              'typical_stress')])
    # current / typical stress level
    stats['current / typical stress'] = stats['current stress'] / stats['typical stress']

    # current alertness
    alert_dict = {'very sluggish': -2,
                  'a little sluggish': -1,
                  'neutral': 0,
                  'a little alert': 1,
                  'very alert': 2}
    stats['alertness'] = pd.Series(index=index,
                                   data=[alert_dict[k.lower()] for k in get_raw_feature(parsed['survey'],
                                                                                        'current_alert')])

    # reported water cups
    stats['water intake'] = pd.Series(index=index, data=[int(c[0]) for c in get_raw_feature(parsed['fitbit'],
                                                                                            'water_cups')])

    # tracked water cups: exclude (values reported by fitbit are unrealistic-- e.g., hundreds of cups of water/day;
    # this could be due to a processing or interpretation issue, or a bug in the fitbit API)
    # reported water cups / tracked water cups: exclude (see above)

    # reported coffee cups
    stats['coffee intake'] = pd.Series(index=index, data=[int(c[0]) for c in get_raw_feature(parsed['survey'],
                                                                                             'coffee_cups')])

    # living setting
    stats['location'] = pd.get_dummies([x.lower() for x in get_raw_feature(parsed['survey'], 'live_setting')])

    # typical job activity level
    job_activity_dict = {'sedentary (e.g., desk job)': 0,
                         'slightly active': 1,
                         'active': 2,
                         'highly active (e.g., heavy lifting)': 3}
    stats['occupation activity level'] = pd.Series(index=index,
                                                   data=[job_activity_dict[k.lower()] for k in
                                                         get_raw_feature(parsed['survey'], 'job_activity')])

    # self-reported exercised today?
    stats['reported exercise today'] = pd.Series(index=index,
                                                 data=[1 if r.lower() == 'yes' else 0 for r in
                                                       get_raw_feature(parsed['survey'], 'exercise_today')])

    # agreement between measured and reported exercise today (null if haven't synced tracker)
    minimum_exercise_min = 1
    tracker_synced = [1 if r.lower() == 'yes' else 0 for r in get_raw_feature(parsed['survey'], 'tracker_sync_today')]
    tracked_exercise = get_tracked_exercise(extract_days_prior(parsed['fitbit'], 1,
                                                               today=get_test_day(parsed['experiment'])),
                                            minimum_exercise_min)
    stats['accurate exercise report'] = pd.Series(index=index,
                                                  data=[x if tracker_synced[i] else np.nan for i, x in
                                                        enumerate([r == a for r, a in
                                                                   zip(stats['reported exercise today'],
                                                                       tracked_exercise)])])

    # self-reported plan to exercise today
    stats['plan to exercise'] = pd.Series(index=index,
                                          data=[1 if p.lower() == 'yes' else 0 for p in
                                                get_raw_feature(parsed['survey'], 'exercise_plan')])

    # self-reported typical exercise frequency
    frequency_dict = {
        '0 days per week': 0,
        '1 day per week': 1,
        '2 days per week': 2,
        '3 days per week': 3,
        '4 days per week': 4,
        '5 days per week': 5,
        '6 days per week': 6,
        '7 days per week': 7,
        '8-14 times per week': 11,
        'more than 14 times per week': 15,
        '2 d&#237;as por semana': 2,
        '4 d&#237;as a la semana': 4,
        '7 d&#237;as a la semana': 7}
    stats['reported exercise frequency'] = pd.Series(index=index,
                                                     data=[frequency_dict[k.lower()] for k in
                                                           get_raw_feature(parsed['survey'], 'exercise_freq')])

    # agreement between measured and reported exercise frequency in the baseline interval
    # compute as ((duration / 7) * observed) / max(reported, 7)
    # note 1: only total active minutes are logged prior 1 day before testing, so we can only know whether *some*
    # exercise occurred on a given day, not whether multiple exercise sessions were performed
    # note 2: if the duration is less than 1 week (min_duration), set agreement to np.nan
    min_duration = 7
    observed_exercise, duration = get_days_exercised(extract_days_prior(parsed['fitbit'], baseline_days,
                                                                        today=get_test_day(parsed['experiment'])),
                                                     minimum_exercise_min)
    stats['reported exercise accuracy'] = pd.Series(index=index,
                                                    data=[(ob * d.days / 7) / np.max([r, 7]) if d.days >= min_duration else np.nan
                                                          for ob, d, r in zip(observed_exercise,
                                                                              duration,
                                                                              stats['reported exercise frequency'])])


    # valence (sentiment) of reported motivation for exercising
    classifier = TextClassifier.load('en-sentiment')

    motivations = get_raw_feature(parsed['survey'], 'exercise_motiv')
    unique_motivations = np.unique(motivations)
    motivation_sentiments = [get_sentiment(x.lower(), classifier) for x in unique_motivations]
    exercise_motivation_dict = {m: s for m, s in zip(unique_motivations, motivation_sentiments)}
    stats['exercise motivation sentiment'] = pd.Series(index=index,
                                                       data=[exercise_motivation_dict[m] for m in motivations])

    # valence (sentiment) of reported motivation for wearing tracker
    motivations = get_raw_feature(parsed['survey'], 'tracker_motiv')
    stats['tracker motivation sentiment'] = pd.Series(index=index,
                                                      data=[get_sentiment(m, classifier) for m in motivations])


    # task understanding
    clarity_dict = {
        'very unclear': -2,
        'unclear': -1,
        'clear': 1,
        'very clear': 2
    }

    # clarity of fitbit instructions
    stats['clarity: fitbit setup'] = pd.Series(index=index,
                                               data=[clarity_dict[c.lower()] for c in
                                                     get_raw_feature(parsed['survey'], 'fitbit_clear')])

    # clarity of free recall instructions
    stats['clarity: free recall (immediate)'] = pd.Series(index=index, data=[clarity_dict[c.lower()] for c in
                                                                             get_raw_feature(parsed['survey'],
                                                                                             'wordlist_clear')])
    # clarity of delayed free recall instructions
    stats['clarity: free recall (delayed)'] = pd.Series(index=index, data=[clarity_dict[c.lower()] for c in
                                                                           get_raw_feature(parsed['survey'],
                                                                                           'delayed_wordlist_clear')])

    # clarity of vocab instructions
    stats['clarity: vocab learning (immediate)'] = pd.Series(index=index, data=[clarity_dict[c.lower()] for c in
                                                                                get_raw_feature(parsed['survey'],
                                                                                                'vocab_clear')])

    # clarity of delayed vocab instructions
    stats['clarity: vocab learning (delayed)'] = pd.Series(index=index, data=[clarity_dict[c.lower()] for c in
                                                                              get_raw_feature(parsed['survey'],
                                                                                              'delayed_vocab_clear')])

    # clarity of spatial task instructions
    stats['clarity: spatial learning (immediate)'] = pd.Series(index=index, data=[clarity_dict[c.lower()] for c in
                                                                                  get_raw_feature(parsed['survey'],
                                                                                                  'spatial_clear')])

    # clarity of movie task instructions
    stats['clarity: naturalistic recall (immediate)'] = pd.Series(index=index,
                                                                  data=[clarity_dict[c.lower()] for c in
                                                                        get_raw_feature(parsed['survey'],
                                                                                        'movie_clear')])

    # clarity of delayed movie task instructions
    stats['clarity: naturalistic recall (delayed)'] = pd.Series(index=index,
                                                                data=[clarity_dict[c.lower()] for c in
                                                                      get_raw_feature(parsed['survey'],
                                                                                      'delayed_movie_clear')])

    # clarity of survey instructions
    stats['clarity: survey'] = pd.Series(index=index, data=[clarity_dict[c.lower()] for c in
                                                            get_raw_feature(parsed['survey'], 'survey_clear')])

    # overall clarity of instructions
    stats['clarity: overall'] = pd.Series(index=index, data=[clarity_dict[c.lower()] for c in
                                                             get_raw_feature(parsed['survey'], 'overall_clear')])

    # self-reported task performance
    difficulty_dict = {
        'very difficult': -2,
        'difficult': -1,
        'medium': 0,
        'easy': 1,
        'very easy': 2
    }

    # difficulty of free recall task
    stats['difficulty: free recall (immediate)'] = pd.Series(index=index,
                                                             data=[difficulty_dict[d.lower()] for d in
                                                                   get_raw_feature(parsed['survey'],
                                                                                   'wordlist_difficult')])

    # difficulty of delayed free recall task
    stats['difficulty: free recall (delayed)'] = pd.Series(index=index,
                                                           data=[difficulty_dict[d.lower()] for d in
                                                                 get_raw_feature(parsed['survey'],
                                                                                 'delayed_wordlist_difficult')])

    # difficulty of vocab task
    stats['difficulty: vocab learning (immediate)'] = pd.Series(index=index,
                                                                data=[difficulty_dict[d.lower()] for d in
                                                                      get_raw_feature(parsed['survey'],
                                                                                      'vocab_difficult')])

    # difficulty of delayed vocab task
    stats['difficulty: vocab learning (delayed)'] = pd.Series(index=index,
                                                              data=[difficulty_dict[d.lower()] for d in
                                                                    get_raw_feature(parsed['survey'],
                                                                                    'delayed_vocab_difficult')])

    # difficulty of spatial task
    stats['difficulty: spatial learning (immediate)'] = pd.Series(index=index,
                                                                  data=[difficulty_dict[d.lower()] for d in
                                                                        get_raw_feature(parsed['survey'],
                                                                                        'spatial_difficult')])

    # difficulty of movie task
    stats['difficulty: naturalistic recall (immediate)'] = pd.Series(index=index,
                                                                     data=[difficulty_dict[d.lower()] for d in
                                                                           get_raw_feature(parsed['survey'],
                                                                                           'movie_difficult')])

    # difficulty of delayed movie task
    stats['difficulty: naturalistic recall (delayed)'] = pd.Series(index=index,
                                                                   data=[difficulty_dict[d.lower()] for d in
                                                                         get_raw_feature(parsed['survey'],
                                                                                         'delayed_movie_difficult')])

    # feedback
    feedback = get_raw_feature(parsed['survey'], 'feedback')

    # number of words of feedback on task
    stats['feedback: number of words'] = pd.Series(index=index,
                                                   data=[len(x.strip().split(' ')) - 1 for x in feedback])

    # average sentiment of feedback on task
    stats['feedback: sentiment'] = pd.Series(index=index,
                                             data=get_sentiment(feedback, classifier))

    x = alt_dict2df(stats)
    return pd.DataFrame(index=parsed['participants'], data=x.values, columns=x.columns)


def get_formatted_data():
    data, participants = load_raw()
    return simplify_dict_list([parse_data(d) for d in data], participants)


def load(recent=7, baseline=30):
    preprocessed_dir = os.path.join(DATA_DIR, 'preprocessed')
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    behavioral_fname = os.path.join(preprocessed_dir, 'behavior.pkl')
    survey_fname = os.path.join(preprocessed_dir, f'survey_{baseline}.pkl')
    fitbit_fname = os.path.join(preprocessed_dir, f'fitbit_{recent}_{baseline}.pkl')

    if not (os.path.exists(behavioral_fname) and
            os.path.exists(survey_fname) and
            os.path.exists(fitbit_fname)):
        parsed_data = get_formatted_data()
    else:
        parsed_data = None

    if not os.path.exists(behavioral_fname):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

        behavioral = behavioral_stats(parsed_data)
        with open(behavioral_fname, 'wb') as f:
            pickle.dump(behavioral, f)

    with open(behavioral_fname, 'rb') as f:
        behavioral = pickle.load(f)


    if not os.path.exists(survey_fname):
        survey = survey_stats(parsed_data, baseline_days=baseline)
        with open(survey_fname, 'wb') as f:
            pickle.dump(survey, f)

    with open(survey_fname, 'rb') as f:
        survey = pickle.load(f)


    if not os.path.exists(fitbit_fname):
        fitbit = fitness_stats(parsed_data, baseline_days=baseline, reference_days=recent)
        with open(fitbit_fname, 'wb') as f:
            pickle.dump(fitbit, f)

    with open(fitbit_fname, 'rb') as f:
        fitbit = pickle.load(f)

    return behavioral, fitbit, survey
