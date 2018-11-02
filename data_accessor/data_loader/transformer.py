from datetime import datetime
from datetime import timedelta

import six
from isoweek import Week

from Settings import *
from utilities import compute_label_encoders


class Transform(object):

    def __init__(self,
                 feature_transforms,
                 label_encoders=None,
                 db_file=None,
                 min_start_date=None,
                 max_end_date=None,
                 target_dates=[],
                 training_transformation=True,
                 testing_Transformation=False,
                 keep_zero_stock_filter=0.0,
                 keep_zero_sale_filter=1.0,
                 stock_threshold=0,
                 output_size=OUTPUT_SIZE,
                 total_input=TOTAL_INPUT,
                 keep_percentage_zero_price=0.0,
                 no_additional_left_zeros=True,
                 no_additional_right_zeros=True,
                 activate_filters=True):
        '''

        :param feature_transforms:
        :param label_encoders:
        :param db_file:
        :param min_start_date:
        :param max_end_date: It is inclusive.
        :param target_dates: list of target dates. These are naturally inclusive dates.
        :param training_transformation: If False, It should output features for production
        '''
        self.keep_zero_stock_filter = keep_zero_stock_filter
        self.keep_zero_sale_filter = keep_zero_sale_filter
        self.keep_zero_price_percentage = keep_percentage_zero_price
        self.stock_threshold = stock_threshold
        self.feature_transforms = feature_transforms
        self.no_additional_left_zeros = no_additional_left_zeros
        self.no_additional_right_zeros = no_additional_right_zeros
        self.testing_Transformation = testing_Transformation
        self.output_size = output_size
        self.total_input = total_input
        self.total_length = self.output_size + self.total_input
        if label_encoders is None:
            assert db_file is not None, 'Path to DB to compute label encoder is not provided'
            self.label_encoders = compute_label_encoders(db_file)

        else:
            self.label_encoders = label_encoders
        self.training_transformation = training_transformation
        if target_dates:
            assert min_start_date is None and max_end_date is None, 'start_dates and mi_start and max_end dates are ' \
                                                                    'mutually exclusive.'
        self.target_dates = []
        if target_dates:
            for dat in target_dates:
                end_date = self.nearest_leading_iso_week(self.convert_to_datetime(dat)) + timedelta(
                    7)  # To make the end_date exclusive as we have with current_date in batch data
                start_date = end_date - timedelta(self.total_input * 7)
                self.target_dates.append((start_date, end_date))
        else:
            self.target_dates = [(
                self.nearest_leading_iso_week(self.convert_to_datetime(min_start_date)),
                self.nearest_leading_iso_week(self.convert_to_datetime(max_end_date)) + timedelta(7),
            )]
        self.activate_filters = activate_filters

    def set_output_size(self, output_size):
        self.target_dates = self.target_dates + timedelta(weeks=output_size - self.output_size)
        self.output_size = output_size
        self.total_length = self.total_input + self.output_size
        self.feature_transforms.set_total_length(self.total_length)

    def convert_to_datetime(self, input_date):
        if isinstance(input_date, six.string_types):
            try:
                return datetime.strptime(input_date, '%Y-%m-%d').date()
            except Exception:
                raise ValueError("Date format should be YYYY-MM-DD")
        else:
            return input_date

    def nearest_leading_iso_week(self, input_date):
        return Week(*input_date.isocalendar()[0:2]).monday()

    def filter_week(self, csku_object, min_start_date, max_end_date):
        first_sale_date = csku_object[FIRST_SALE_DATE]
        current_date = csku_object[CURRENT_DATE]

        if self.training_transformation:
            if current_date <= min_start_date:
                return [], ('', '')
            if first_sale_date >= max_end_date:
                return [], ('', '')
            start_date = max(min_start_date, first_sale_date)
            end_date = min(max_end_date, current_date)  # End date is always exclusive
        else:
            end_date = current_date
            start_date = max(current_date - timedelta(self.total_input * 7),
                             first_sale_date)

        num_weeks = (end_date - start_date).days // 7
        start_idx = (start_date - first_sale_date).days // 7
        end_idx = num_weeks + start_idx

        return range(start_idx, end_idx), (start_date, end_date)

    def filter_out_low_stock(self, feature_dictionary, target_index, threshold):
        return feature_dictionary[STOCK][target_index] <= threshold

    def filter_out_low_sales(self, feature_dictionary, first_target_week, threshold):
        return sum(feature_dictionary[SALES_MATRIX][:, first_target_week]) <= threshold

    def filter_out_zero_price(self, feature_dictionary):
        return feature_dictionary[BLACK_PRICE_INT][0] <= 0

    def __call__(self, csku_object, *args, **kwargs):
        transformed_samples = []
        for min_start_date, max_end_date in self.target_dates:
            idx_range, (start_period, end_period) = self.filter_week(csku_object, min_start_date, max_end_date)

            if not idx_range:
                continue
            num_weeks = len(idx_range)
            sufficient_length = self.total_length if self.training_transformation else self.total_input

            num_zeros = sufficient_length - num_weeks if num_weeks < sufficient_length else 0
            num_zeros_right = 0
            if self.training_transformation:
                if not self.no_additional_left_zeros:
                    num_zeros = max(num_zeros, self.total_length - 1 - self.output_size)
                if not self.no_additional_right_zeros:
                    num_zeros_right = self.output_size - 1
            csku_object = self.feature_transforms.enrich_csku(csku_object, self.training_transformation)
            arguments = {'num_zeros': num_zeros, 'num_zeros_right': num_zeros_right, 'num_weeks': num_weeks,
                         'start_period': start_period,
                         'end_period': end_period, 'label_encoders': self.label_encoders}

            feature_dictionary = self.feature_transforms.transform(csku_object, idx_range, **arguments)
            if self.training_transformation:
                chunk_data = self.create_chunk_of_samples(feature_dictionary)
                transformed_samples.extend(chunk_data)
            else:
                if POSITION_ENCODING in feature_dictionary:
                    feature_dictionary[POSITION_ENCODING] = self.feature_transforms.positional_encoding_values
                transformed_samples.extend(self.feature_transforms.to_final_format_prediction(feature_dictionary))
        return transformed_samples

    def create_chunk_of_samples(self, feature_dictionary):
        num_countries, num_weeks = feature_dictionary[SALES_MATRIX].shape
        num_window_shifts = int(np.floor((num_weeks - self.total_length) / float(WINDOW_SHIFT))) + 1
        samples = []
        for slide_i in range(num_window_shifts):
            if slide_i < self.output_size and not self.no_additional_right_zeros:
                loss_mask = np.concatenate([np.ones(slide_i + 1), np.zeros(self.output_size - slide_i - 1)])
            else:
                loss_mask = np.ones(self.output_size)
            if slide_i >= 1 and self.testing_Transformation:
                return samples
            first_target_week_idx = num_weeks - slide_i * WINDOW_SHIFT - self.output_size
            if self.activate_filters and self.filter_out_low_stock(feature_dictionary, first_target_week_idx,
                                                                   self.stock_threshold):
                if np.random.rand() >= self.keep_zero_stock_filter:
                    continue
            if self.activate_filters and self.filter_out_low_sales(feature_dictionary, first_target_week_idx, 0):
                if np.random.rand() >= self.keep_zero_sale_filter:
                    continue

            if self.activate_filters and self.filter_out_zero_price(feature_dictionary):
                if np.random.rand() >= self.keep_zero_price_percentage:
                    continue

            selected_range = range(first_target_week_idx - self.total_input, first_target_week_idx + self.output_size)
            sample = self.feature_transforms.to_final_format_training(feature_dictionary, selected_range,
                                                                      self.activate_filters)
            if sample != []:
                samples.append([sample, loss_mask])  # NUM_SAMPLES x TOTAL_LENGTH x NUM_FEAT
        return samples
