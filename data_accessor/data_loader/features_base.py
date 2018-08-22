from abc import ABCMeta, abstractmethod


class FeaturesBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, feature_descriptions):
        self.feature_descriptions = feature_descriptions

    @abstractmethod
    def to_feature_parameter_format(self, csku_object):
        pass

    @abstractmethod
    def extract(self, csku_object, key, idx_range):
        pass

    @abstractmethod
    def transform(self,csku_object, idx_range, **kwargs):
        pass

    @abstractmethod
    def enrich_csku(self, csku_object, training_transformation=True):
        return csku_object

    @abstractmethod
    def to_final_format_training(self, feature_dictionary,idx_range,activate_filters):
        pass

    @abstractmethod
    def to_final_format_prediction(self, feature_dictionary):
        pass
