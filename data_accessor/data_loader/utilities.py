import cPickle as pickle
import sqlite3
from datetime import timedelta
import types

from Settings import *
import codecs
import json
from numba import jit

@jit()
def zero_padder(x, num_zeros, num_zeros_right, global_or_international, **kwargs):
    size = x.shape
    if num_zeros == 0 and num_zeros_right == 0:
        out = x
    elif global_or_international == DYNAMIC_GLOBAL:
        out = np.concatenate([[0] * num_zeros, x, [0] * num_zeros_right])
    else:
        out = np.concatenate([np.zeros((size[0], num_zeros)), x, np.zeros((size[0], num_zeros_right))], axis=1)
    return out


def encode_strings(value, feature, label_encoders, **kwarg):
    if isinstance(value, types.ListType):
        return [
            label_encoders[feature][val] if val in label_encoders[feature] else label_encoders[feature][MISSING_VALUE]
            for val in value]
    else:
        return label_encoders[feature][value] if value in label_encoders[feature] else label_encoders[feature][
            MISSING_VALUE]


def to_string(value, **kwarg):
    if not isinstance(value, types.StringTypes):
        return str(int(value))
    else:
        return value


def iso_week_generate(csku_object, training_transformation):
    start_date = csku_object[FIRST_SALE_DATE]
    current_date = csku_object[CURRENT_DATE]
    num_weeks = (current_date - start_date).days // 7
    future_weeks = 0 if training_transformation else OUTPUT_SIZE
    iso_week_seq = [(start_date + timedelta(idx * 7)).isocalendar()[1] for idx in range(num_weeks + future_weeks)]
    return np.array(iso_week_seq)


def get_supplier_stock_uplift(past_stock, past_sales, past_returns):
    """ Compute what we estimate as a supplier stock uplift caused by buying stock."""
    # slicing with [0:-1] is cutting away the last element
    uplift = np.maximum(np.diff(past_stock) + past_sales[0:-1] - past_returns[1:], 0)
    return np.concatenate([uplift, [0]])

@jit()
def log_transform(x, is_transform, **kwargs):
    return np.log(x + 1.0) if is_transform else x


def add_country(csku_object):
    csku_object[COUNTRY] = [str(i) for i in range(NUM_COUNTRIES)]
    return csku_object


def add_cgs(csku_object):
    for idx, cg in enumerate(csku_object[COMMODITY_GROUPS]):
        csku_object[u'CG{i}'.format(i=idx + 1)] = cg
    return csku_object

@jit()
def high_dimensional_harmonic(x, **kwargs):
    size = kwargs[SIZE]
    x1 = np.repeat(x[:, None], repeats=size, axis=1) * 2 * np.pi / 52.0 * ((2 * np.arange(size) + 1.0) / size)
    # SIZE: TOTAL_LENGTH x size
    encoded_iso_week = np.concatenate([np.sin(x1[:, 0:size:2]), np.cos(x1[:, 1:size:2])], axis=1)

    return encoded_iso_week


def iso_week_padding(x, start_period, num_zeros, num_zeros_right, **kwarg):
    if num_zeros == 0: return x
    zero_padded_start_date = start_period - timedelta(num_zeros * 7)
    iso_week_seq = []
    for idx in range(num_zeros):
        week_date = zero_padded_start_date + timedelta(idx * 7)
        iso_week_seq.append(week_date.isocalendar()[1])
    return np.concatenate([iso_week_seq, x, np.zeros(num_zeros_right)], axis=0)


def compute_label_encoders(db_file, include_country=True, convert_to_string=True, chunk_size=100):
    connection = sqlite3.connect(db_file)
    conn_db = connection.cursor()
    count_num_rows = 'SELECT max(_ROWID_) FROM data'
    conn_db.execute(count_num_rows)
    num_samples = conn_db.fetchall()[0][0]
    row_indices = range(1, num_samples + 1)
    query = lambda row_indices: 'SELECT dictionary FROM data WHERE rowid IN {row_indices}'.format(
        row_indices=tuple(row_indices))
    # TODO
    num_chunks = num_samples // chunk_size + 1
    # num_chunks = 1
    encode_features = [feature for feature, description in FEATURE_DESCRIPTIONS.iteritems() if
                       LABEL_ENCODING in description[TRANSFORMATION]]
    unique_values = {feature: {MISSING_VALUE} for feature in encode_features}
    for i in range(num_chunks):
        selected_samples = tuple(row_indices[i * chunk_size:(i + 1) * chunk_size])
        select_query = query(selected_samples)
        conn_db.execute(select_query)
        rows = conn_db.fetchall()
        for row in rows:
            csku_object = (pickle.loads(str(row[0])))
            csku_object = add_cgs(csku_object)
            for feature, description in FEATURE_DESCRIPTIONS.iteritems():
                if LABEL_ENCODING in description[TRANSFORMATION]:
                    value = csku_object.get(feature, MISSING_VALUE)
                    if isinstance(value, types.ListType):
                        for val in value: unique_values[feature].add(val)
                    else:
                        unique_values[feature].add(value)

    label_encoders = {}
    for feature, unique_vals in unique_values.iteritems():
        label_encoders[feature] = dict(
            (value, idx) if not convert_to_string else (to_string(value), idx)
            for idx, value in enumerate(unique_vals)
        )
    if include_country:
        label_encoders[COUNTRY] = {val: val for val in range(NUM_COUNTRIES)}
    label_encoders[ISO_WEEK_SEQ] = {ind: val for ind, val in enumerate(range(1, 54))}

    return label_encoders


def load_label_encoder(file_path):
    with codecs.open(file_path, 'r', 'utf-8') as categorical_feature_mapping_file:
        categorical_feature_mapping_json = categorical_feature_mapping_file.read().strip()
    categorical_feature_mapping = json.loads(categorical_feature_mapping_json)
    return categorical_feature_mapping


def save_label_encoder(label_encoders, file_path):
    with codecs.open(file_path, 'w', 'utf-8') as categorical_feature_mapping_file:
        categorical_feature_mapping_file.write(json.dumps(label_encoders))


def append_lists(inputs):
    return sum(inputs, [])
