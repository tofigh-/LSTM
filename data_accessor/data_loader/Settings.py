from collections import OrderedDict
import numpy as np

NUM_COUNTRIES = 14

### Feature Names

SOLD_OUT_DATE = u'Sold Out Target Date'  # useless as it is only for the current date stuff.
BLACK_PRICE_INT = u'Black Price INT'
SIZE = u"size"
RETURNS_TIMES_MATRIX = u'INT Returns Times Matrix'  # num of returns in a given week
RETURNS_MATRIX = u'INT Returns Matrix'  # num of items eventually return of the sales of the given week
SALES_MATRIX = u'INT Sales Matrix'
GLOBAL_SALE = u'Global Sale'
DISCOUNT_MATRIX = u'INT Discount Matrix'
STOCK = u'Past Stock'
STOCK_UPLIFT = u'Stock Uplift'
BINARY_SEASON_FLAGS = u'Past Binary Season Flags'

CONFIG_SKU = u'Config SKU'
BRAND = u'Brand'
HDG_INDEX = u'HDG Index'
COMMODITY_GROUPS = u'Commodity Groups'
SEASON_TYPE = u'Season Type'
CATEGORY = u'Category'
CG1, CG2, CG3, CG4, CG5 = u'CG1', u'CG2', u'CG3', u'CG4', u'CG5'
COUNTRY = u'Country'

CURRENT_DATE = u'Current Date'
FIRST_SALE_DATE = u'First Sale Date'

PAST_RETURNS = u'Past Returns'  # sum of returns no use I guess.
ISO_WEEK_SEQ = u'ISO Week Seq'

TYPE = u"Global or International"
TRANSFORMATION = u"Transformation"
DYNAMIC_INT = u"Dynamic International"
DYNAMIC_GLOBAL = u"Dynamic Global"
STATIC_INT = u"Static International"
STATIC_GLOBAL = u"Static Global"
LOG_TRANSFORM = u"LOG_TRANSFORM"
LABEL_ENCODING = u"Embedding"
ZERO_PAD = u"Zero Pad"
ISO_WEEK_PADDING = u'ISO Week Padding'
TO_STRING = u'To String'
HIGH_DIMENSIONAL_SIN = u"High Dimensional Sin"
EMBEDDING_SIZE = u'Embedding Size'
EMBEDDING_MAX = u'Embedding Max Integer'

# Specific to prediction time
FUTURE_BINARY_SEASON_FLAGS = u'Future Binary Season Flags'
INT_FUTURE_DISCOUNT = u'INT Future Discounts'
STOCK_UPLIFT_FUTURE = u'Stock Uplift Vector'

## Boolean Switches
IS_LOG_TRANSFORM = True
FILTER_LOW_STOCK = True
FILTER_ZERO_PRICE = True
FILTER_LOW_SALE = True
SUM_LOSS = False
RESUME = False
## Variables
ENCODER_CHECKPOINT = u'Encoder Checkpoint'
ENCODER_DECODER_CHECKPOINT = u'Encoder Decoder Checkpoint'
OPTIMIZER = u'Optimizer'
STATE_DICT = u'State Dict'
FUTURE_DECODER_CHECKPOINT = u'Future Decoder Checkpoint'
MISSING_VALUE = u'-1'
PAST_UNKNOWN_RETURN_LENGTH = 0
OUTPUT_SIZE = 5
PAST_KNOWN_LENGTH = 52
TOTAL_LENGTH = PAST_KNOWN_LENGTH + PAST_UNKNOWN_RETURN_LENGTH + OUTPUT_SIZE
TOTAL_RETURN_INPUT = PAST_KNOWN_LENGTH
TOTAL_INPUT = PAST_KNOWN_LENGTH + PAST_UNKNOWN_RETURN_LENGTH
WINDOW_SHIFT = 1
TEST_BATCH_SIZE = 256
BATCH_SIZE = 256
BN_MOMENTUM = 0.0001
GRADIENT_CLIP = 5
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 512
NUM_LAYER = 1
NUM_BATCH_SAVING_MODEL = 10000
BI_DIRECTIONAL = True
RNN_DROPOUT = 0.5
ENCODER_WEIGHT_DECAY = 0.00001
DECODER_WEIGHT_DECAY = 0.0
TEST_STOCK_THRESHOLD = 0
TRAIN_STOCK_THRESHOLD = 2
TRAIN_ZERO_SALE_PERCENTAGE = 0.1
TEST_ZERO_SALE_PERCENTAGE = 1.0
UPDATE_LOSS_AT_BATCH_NUM = 2000
use_future_unknown_estimates = True
list_l2_loss_countries = list(range(0, 14))
list_l1_loss_countries = list(range(0, 14))
cache_training = True
USE_ALREADY_CACHED = False
to_print_variables = ['list_l2_loss_countries', 'list_l1_loss_countries', 'TEST_ZERO_SALE_PERCENTAGE',
                      'TRAIN_ZERO_SALE_PERCENTAGE', 'WINDOW_SHIFT',
                      'TRAIN_STOCK_THRESHOLD', 'TEST_STOCK_THRESHOLD', 'use_future_unknown_estimates', 'RESUME',
                      'ENCODER_WEIGHT_DECAY', 'RNN_DROPOUT', 'BI_DIRECTIONAL', 'NUM_BATCH_SAVING_MODEL',
                      'NUM_LAYER',
                      'HIDDEN_SIZE', 'LEARNING_RATE', 'GRADIENT_CLIP', 'BATCH_SIZE', 'TEST_BATCH_SIZE',
                      'TOTAL_INPUT', 'TOTAL_LENGTH', 'PAST_KNOWN_LENGTH', 'OUTPUT_SIZE']
# Note that Country is a Static international feature
static_global_features = [BRAND, HDG_INDEX, CG1, CG2, CG3, CG4, CG5, SEASON_TYPE, CATEGORY]

FEATURE_DESCRIPTIONS = OrderedDict([
    (SALES_MATRIX, {TYPE: DYNAMIC_INT, TRANSFORMATION: [LOG_TRANSFORM, ZERO_PAD]}),
    (BLACK_PRICE_INT, {TYPE: STATIC_INT, TRANSFORMATION: [LOG_TRANSFORM]}),
    # COUNTRY: {TYPE: STATIC_INT, TRANSFORMATION: [LABEL_ENCODING]},
    # RETURNS_TIMES_MATRIX: {TYPE: DYNAMIC_INT, TRANSFORMATION: [LOG_TRANSFORM, ZERO_PAD]},
    # RETURNS_MATRIX: {TYPE: DYNAMIC_INT, TRANSFORMATION: [LOG_TRANSFORM, ZERO_PAD]},
    (DISCOUNT_MATRIX, {TYPE: DYNAMIC_INT, TRANSFORMATION: [ZERO_PAD]}),
    (GLOBAL_SALE, {TYPE: DYNAMIC_GLOBAL, TRANSFORMATION: [LOG_TRANSFORM, ZERO_PAD]}),
    (ISO_WEEK_SEQ, {TYPE: DYNAMIC_GLOBAL, TRANSFORMATION: [ISO_WEEK_PADDING, HIGH_DIMENSIONAL_SIN], SIZE: 10}),
    (BINARY_SEASON_FLAGS, {TYPE: DYNAMIC_GLOBAL, TRANSFORMATION: [ZERO_PAD]}),
    (STOCK_UPLIFT, {TYPE: DYNAMIC_GLOBAL, TRANSFORMATION: [LOG_TRANSFORM, ZERO_PAD]}),
    (STOCK, {TYPE: DYNAMIC_GLOBAL, TRANSFORMATION: [LOG_TRANSFORM, ZERO_PAD]})
])

for feature in static_global_features: FEATURE_DESCRIPTIONS[feature] = {TYPE: STATIC_GLOBAL,
                                                                        TRANSFORMATION: [LABEL_ENCODING]}
FEATURE_DESCRIPTIONS[HDG_INDEX][TRANSFORMATION].insert(0, TO_STRING)
# It particularly specifies the order that embedded features appear in the concatenated embedded feature vector
embedded_features = [feature for feature, description in FEATURE_DESCRIPTIONS.iteritems()
                     if LABEL_ENCODING in description[TRANSFORMATION]
                     ]
EMBEDDING_SIZE_FOR_ALL = 10
embedding_descriptions = \
    {
        # COUNTRY: {EMBEDDING_SIZE: int(14), EMBEDDING_MAX: 15},
        CG1: {EMBEDDING_SIZE: int(EMBEDDING_SIZE_FOR_ALL), EMBEDDING_MAX: None}
    }

for feature in embedded_features:
    if feature not in embedding_descriptions:
        embedding_descriptions[feature] = {EMBEDDING_SIZE: EMBEDDING_SIZE_FOR_ALL, EMBEDDING_MAX: None}

import numpy as np

feature_indices = dict()
ind = 0
for feature, descript in FEATURE_DESCRIPTIONS.iteritems():
    if descript[TYPE] == STATIC_INT or descript[TYPE] == DYNAMIC_INT:
        feature_indices[feature] = np.arange(ind, ind + NUM_COUNTRIES)
        ind = ind + NUM_COUNTRIES
    elif SIZE in descript.keys():
        feature_indices[feature] = np.arange(ind, ind + descript[SIZE])
        ind = ind + descript[SIZE]
    else:
        feature_indices[feature] = [ind]
        ind = ind + 1

# feature_indices = {feature: ind for ind, feature in enumerate(FEATURE_DESCRIPTIONS.iterkeys())}
numeric_feature_indices = np.concatenate(
    [feature_indices[feature] for feature, description in FEATURE_DESCRIPTIONS.iteritems()
     if LABEL_ENCODING not in description[TRANSFORMATION]])

assert sorted(embedding_descriptions.keys()) == sorted(embedded_features)

embedding_feature_indices = np.concatenate([feature_indices[feature] for feature in embedded_features])
