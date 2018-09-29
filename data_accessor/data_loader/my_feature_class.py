from features_base import FeaturesBase
from Settings import *
from utilities import add_cgs, iso_week_generate, get_supplier_stock_uplift, log_transform, zero_padder, \
    iso_week_padding, encode_strings, to_string, add_country, high_dimensional_harmonic
import numpy as np
from numpy import inf


class MyFeatureClass(FeaturesBase):
    def __init__(self, feature_descriptions, low_sale_percentage):
        super(MyFeatureClass, self).__init__(feature_descriptions)
        self.transformations = {LOG_TRANSFORM: log_transform, LABEL_ENCODING: encode_strings,
                                ZERO_PAD: zero_padder, ISO_WEEK_PADDING: iso_week_padding, TO_STRING: to_string
            , HIGH_DIMENSIONAL_SIN: high_dimensional_harmonic}
        self.low_sale_percentage = low_sale_percentage

    def to_feature_parameter_format(self, csku_object):
        return csku_object

    def extract(self, csku_object, key, idx_range):
        global_or_international = self.feature_descriptions[key][TYPE]
        if global_or_international == DYNAMIC_GLOBAL:
            return csku_object[key][idx_range]
        if global_or_international == DYNAMIC_INT:
            return csku_object[key][:, idx_range]
        if global_or_international == STATIC_INT:
            return csku_object.get(key, np.zeros(NUM_COUNTRIES))
        if global_or_international == STATIC_GLOBAL:
            return csku_object.get(key, MISSING_VALUE)

    def enrich_csku(self, csku_object, training_transformation=True):
        iso_week_seq = iso_week_generate(csku_object, training_transformation)
        csku_object[ISO_WEEK_SEQ] = iso_week_seq
        csku_object = add_cgs(csku_object)
        csku_object[GLOBAL_SALE] = np.sum(csku_object[SALES_MATRIX], axis=0)
        csku_object = add_country(csku_object)
        csku_object[STOCK_UPLIFT] = np.append(0,np.maximum(csku_object[STOCK_UPLIFT], 0))
        csku_object[STOCK_UPLIFT][csku_object[STOCK_UPLIFT] == inf] = 0
        csku_object[STOCK_UPLIFT][csku_object[STOCK_UPLIFT] == -inf] = 0
        csku_object[STOCK][csku_object[STOCK] == inf] = 0
        csku_object[STOCK][csku_object[STOCK] == -inf] = 0

        if training_transformation:
            return csku_object

        csku_object[STOCK_UPLIFT] = np.concatenate(
            [
                get_supplier_stock_uplift(csku_object[STOCK],
                                          np.sum(csku_object[SALES_MATRIX], 0),
                                          np.sum(csku_object[RETURNS_TIMES_MATRIX], 0)),
                csku_object[STOCK_UPLIFT_FUTURE]
            ], 0)

        csku_object[BINARY_SEASON_FLAGS] = np.concatenate([csku_object[BINARY_SEASON_FLAGS],
                                                           csku_object[FUTURE_BINARY_SEASON_FLAGS]])
        csku_object[DISCOUNT_MATRIX] = np.concatenate([csku_object[DISCOUNT_MATRIX],
                                                       csku_object[INT_FUTURE_DISCOUNT]],
                                                      axis=1)
        return csku_object

    def transform(self, csku_object, idx_range, **kwargs):
        feature_dictionary = {}
        _, num_available_weeks = csku_object[SALES_MATRIX].shape

        for feature, description in self.feature_descriptions.iteritems():

            transformation_arguments = {'is_transform': IS_LOG_TRANSFORM, 'num_zeros': kwargs['num_zeros'],
                                        'global_or_international': description[TYPE], 'feature': feature,
                                        'start_period': kwargs['start_period'],
                                        'label_encoders': kwargs['label_encoders']}
            if SIZE in description.keys():
                transformation_arguments[SIZE] = description[SIZE]

            feature_dictionary[feature] = self.extract(csku_object, feature, idx_range)
            for transformation in description[TRANSFORMATION]:
                feature_dictionary[feature] = self.transformations[transformation](feature_dictionary[feature],
                                                                                   **transformation_arguments)
        return feature_dictionary

    def filter_zero_price(self, country_samples):
        final_samples = [country_sample for country_sample in country_samples if
                         country_sample[-1, feature_indices[BLACK_PRICE_INT]] > 0]
        return final_samples

    def filter_low_sales(self, country_samples, threshold):
        final_samples = [country_sample for country_sample in country_samples if
                         country_sample[-1, feature_indices[SALES_MATRIX]] > threshold]
        return final_samples

    def independent_country_samples(self, feature_dictionary, idx_range, activate_filters):
        feature_seq = []
        for feature, description in self.feature_descriptions.iteritems():
            feature_value = self.extract(feature_dictionary, feature, idx_range)
            if description[TYPE] is STATIC_GLOBAL:
                f = np.ones((NUM_COUNTRIES, TOTAL_LENGTH)) * feature_value
            if description[TYPE] is STATIC_INT:
                f = np.repeat(np.array(feature_value)[:, None], TOTAL_LENGTH, axis=1)
            if description[TYPE] is DYNAMIC_GLOBAL:
                f = np.repeat(np.array(feature_value)[None, :], NUM_COUNTRIES, axis=0)
            if description[TYPE] is DYNAMIC_INT:
                f = feature_value
            feature_seq.append(f)
            # 0 : 40
            # 1 : 64
            # 2 : 83
            # 3 : 68
            # 4 : 112
            # 5 : 78
            # 6 : 50
            # 7 : 88
            # 8 : 105
            # 9 : 72
            # 10 : 96
            # 11 : 99
            # 12 : 96
            # 13 : 97
        feature_seq = list(np.moveaxis(np.array(feature_seq), source=[0, 1, 2],
                                       destination=[2, 0, 1])[4, :, :][None, :,
                           :])  # NUM_COUNTRY xTOTAL_LENGTH x NUM_FEAT
        if FILTER_ZERO_PRICE and activate_filters:
            feature_seq = self.filter_zero_price(feature_seq)

        if FILTER_LOW_SALE and activate_filters and np.random.rand() > self.low_sale_percentage:
            feature_seq = self.filter_low_sales(feature_seq, threshold=0)
        return feature_seq

    def dependent_country_samples(self, feature_dictionary, idx_range, activate_filters):
        feature_seq = []
        if FILTER_ZERO_PRICE and activate_filters:
            if feature_dictionary[BLACK_PRICE_INT][-1] <= 0.0:
                return feature_seq

        for feature, description in self.feature_descriptions.iteritems():
            feature_value = self.extract(feature_dictionary, feature, idx_range)
            if description[TYPE] is STATIC_GLOBAL:
                f = np.ones((1, TOTAL_LENGTH)) * feature_value
            if description[TYPE] is STATIC_INT:
                f = np.repeat(np.array(feature_value)[:, None], TOTAL_LENGTH, axis=1)
            if description[TYPE] is DYNAMIC_GLOBAL:
                if SIZE not in description.keys():
                    f = np.array(feature_value)[None, :]
                else:
                    f = feature_value.transpose()
            if description[TYPE] is DYNAMIC_INT:
                f = feature_value
            feature_seq.append(f)

        feature_seq = [np.concatenate(feature_seq).transpose()]  # TOTAL_LENGTH x  NUM_FEAT

        return feature_seq

    def to_final_format_training(self, feature_dictionary, idx_range, activate_filters):
        return self.dependent_country_samples(feature_dictionary, idx_range, activate_filters)

    def to_final_format_prediction(self, feature_dictionary):
        pass
