from __future__ import annotations
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from dataclasses import dataclass
import pickle
import time

@dataclass
class FilterSQUID:
    attribute_name: str
    value: tuple
    association_strength: int
    is_categorical: bool
    
    def get_selectivity(self, data):
        selectivity = 1 
        size_filtered = 0
        
        if self.is_categorical:
            size_filtered = data.query(f'{self.attribute_name} == "{self.value[0]}"').shape[0]
        else:
            size_filtered = data.query(f'{self.attribute_name} >= {self.value[0]} and {self.attribute_name} <= {self.value[1]}').shape[0]
        
        selectivity = size_filtered/data.shape[0]
        
        return selectivity
    
    def domain_coverage(self, attribute_domain:list):
        coverage = 0.0
        
        if self.is_categorical:
            coverage = len(set(attribute_domain) & set(self.value))/len(set(attribute_domain) | set(self.value))
        else:
            minimum_value = min(attribute_domain)
            maximum_value = max(attribute_domain)
            coverage = (self.value[1] - self.value[0])/(maximum_value - minimum_value)
        
        return coverage    

@dataclass
class SemanticContextSQUID:
    attribute_name: str
    value: tuple
    association_strength: int
    example_size: int
    is_categorical: bool

class AlphaDatabaseInterface:
    def __init__(self, data:pd.DataFrame, base_query:str):
        if len(base_query) > 0:
            self.base_query_result = data.query(base_query)
        else:
            self.base_query_result = data.copy(deep=True)
        
    def extract_semantic_contexts_from(self, examples:pd.DataFrame):
        example_set_size = examples.shape[0]
        # Each context is a tuple (attribute_name, value_list, association_strength, is_categorical)
        semantic_contexts = []

        for col in self.base_query_result.columns:
            if self.base_query_result[col].dtype == 'O':
                count_values = examples[col].value_counts().to_dict()
                context = [SemanticContextSQUID(col, (value), None, example_set_size,  True) for value, count in count_values.items()
                              if count == example_set_size]
                context = sorted(context)
                semantic_contexts.extend(context)
            else:
                unique_values = sorted(examples[col].unique())
                minimum = unique_values[0]
                maximum = unique_values[-1]
                context = SemanticContextSQUID(col, (minimum, maximum), None, example_set_size, False)
                semantic_contexts.append(context)

        return semantic_contexts

@dataclass
class QueryAbductionSQUID:
    # hyperparamets used in the original work
    base_prior_rho: float = 0.1
    coverage_gamma: int = 2
    association_strength_threshold: int = 5
    skewness_threshold:float = 2.0
    domain_coverage_eta: float = 100
    
    def get_minimum_valid_filters(self, semantic_contexts:list[SemanticContextSQUID]):
        return [FilterSQUID(attribute_name=c.attribute_name, value=c.value, association_strength=c.association_strength, is_categorical=c.is_categorical) for c in semantic_contexts]
    
    def selectivity_impact(self, ftr:FilterSQUID, attribute_domain):
        domain_coverage = ftr.domain_coverage(attribute_domain)
        return 1/(max(1, domain_coverage/self.domain_coverage_eta)**self.coverage_gamma)
    
    def association_strength_impact(self, ftr:FilterSQUID):
        return int(ftr.association_strength == None or ftr.association_strength > self.association_strength_threshold)
    
    def outlier_impact(self, ftr:FilterSQUID, ftr_family:list[FilterSQUID]):
        return 1 # TODO: Implement decision for derived filters
    
    def compute_filter_include_prior_probability(self, ftr:FilterSQUID, minimal_valid_filters:list[FilterSQUID], data:pd.DataFrame):
        ftr_family = [f for f in minimal_valid_filters if f.attribute_name == ftr.attribute_name]
        
        base = self.base_prior_rho 
        sel = self.selectivity_impact(ftr, data[ftr.attribute_name].values)        
        astr = self.association_strength_impact(ftr) * self.outlier_impact(ftr, ftr_family)
        outlr = self.outlier_impact(ftr, ftr_family)
        
        proba = base * sel * astr * outlr
        
        return  proba
    
    def abduct_query(self, examples:pd.DataFrame, alpha_database:AlphaDatabaseInterface):
        semantic_contexts = alpha_database.extract_semantic_contexts_from(examples)
        minimal_valid_filters = self.get_minimum_valid_filters(semantic_contexts)
        
        used_filters = []
        
        for ftr in minimal_valid_filters:
            ftr_probability = self.compute_filter_include_prior_probability(ftr, minimal_valid_filters, alpha_database.base_query_result)
            no_ftr_probability = 1 - ftr_probability
            context_given_ftr_probability = 1
            context_given_no_ftr_probability = ftr.get_selectivity(alpha_database.base_query_result)**examples.shape[0]
            
            include_ftr_prob = ftr_probability * context_given_ftr_probability
            exclude_ftr_prob = no_ftr_probability * context_given_no_ftr_probability
            
            if include_ftr_prob > exclude_ftr_prob:
                used_filters.append(ftr)
        
        predicates = []
        
        for ftr in used_filters:
            if ftr.is_categorical:
                predicates.append(f'{ftr.attribute_name} in ["{ftr.value}"]')
            else:
                predicates.append(f'{ftr.attribute_name} >= {ftr.value[0]} and {ftr.attribute_name} <= {ftr.value[1]}')
            
        query = ' and '.join(predicates)
        return query

def prepare_squid(data, db_name):
    
    # call the alpha database construction
    alpha_database = AlphaDatabaseInterface(data=data, base_query='')

    # for debugging
    with open("data/squid/alpha_db_"+ db_name +".pickle", "wb+") as fp:
        pickle.dump(alpha_database, fp)

def run_squid(examples = None, db_name = None):
    
    # load the precomputed alpha db
    with open("data/squid/alpha_db_"+str(db_name)+".pickle", "rb") as fp:
        alpha_database = pickle.load(fp)

    # measure time
    start_time = time.time()
    
    # SQuID function calls
    squid = QueryAbductionSQUID()
    query = squid.abduct_query(examples, alpha_database)
    result = alpha_database.base_query_result.query(query)

    # stop timing
    stop_time = time.time()

    return [result, stop_time-start_time]