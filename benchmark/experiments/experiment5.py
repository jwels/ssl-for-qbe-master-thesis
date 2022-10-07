from algorithms.decision_tree import run_decision_tree_ssl
from algorithms.random_forest import run_random_forest_ssl
from algorithms.gradient_boosting import run_gradient_boosting_ssl
import pandas as pd
from helper_functions import calculate_f1_score, translate_query_for_onehot_table, build_base_query


# Run fifth experiment: Semi-Supervised Learning, unbalanced Dataset, with Base Query extension for DT, RF, GB
def run_experiment_5(query_set, datasets, datasets_onehot):

    print("----------------------------------------")
    print("Experiment 5: Semi-supervised Learning, unbalanced Dataset, with Base Query extension")
    print("----------------------------------------")

    # construct result dataframe
    benchmark = pd.DataFrame(columns=["algorithm", "dataset", "query", "difficulty", "f1_score", "precision", "recall", "time"])

    # benchmark main loop over query set
    for idx, query in query_set.iterrows():

        # start iteration, get ground truth and input examples
        print("----- Query " + str(idx+1) + "/"+ str(len(query_set)) +" -------")
        # choose the right dataset for the query
        if(query["database"]=="adult"):
            dataset = datasets[0]
            dataset_onehot = datasets_onehot[0]
        if(query["database"]=="movies"):
            dataset = datasets[1]
            dataset_onehot = datasets_onehot[1]
        if(query["database"]=="stars"):
            dataset = datasets[2]
            dataset_onehot = datasets_onehot[2]
        ground_truth = dataset.query(str(query["query"]))
        example_size = min(10, int(ground_truth.shape[0] * 0.5)+1)
        examples = ground_truth.sample(example_size, random_state=0)        
        # draw neg. example set
        neg_query = "not (" + str(query["query"]) + ")"
        neg_ground_truth = dataset.query(neg_query)
        neg_examples = neg_ground_truth.sample(example_size, random_state=0)
        # list columns in the query. In a normal QBE setting these would be known by the user provided example columns.
        base_query_column_subset = [word for word in examples.columns.values if word in str(query["query"])]
        # construct base query on this column subset
        base_query = build_base_query(example=examples[base_query_column_subset])
        neg_base_query = build_base_query(example=neg_examples)
        # get extended example set using the constructed base query
        query_for_onehot = translate_query_for_onehot_table(query["query"])
        base_query_for_onehot = translate_query_for_onehot_table(base_query)
        neg_base_query_for_onehot = translate_query_for_onehot_table(neg_base_query)
        ground_truth_onehot = dataset_onehot.query(query_for_onehot)
        extended_examples_onehot = dataset_onehot.query(base_query_for_onehot).copy()        
        extended_examples_onehot_neg = dataset_onehot.query(neg_base_query_for_onehot).copy()
        # get the "unlabeled" data by droping the randomly drawn entries from the ground truth, if existing
        gt_rest = dataset_onehot.drop(extended_examples_onehot.index, errors = "ignore")
        gt_rest = gt_rest.drop(extended_examples_onehot_neg.index, errors = "ignore")

        # set the labels
        extended_examples_onehot["label"] = 1
        extended_examples_onehot_neg["label"] = 0
        # set to -1 to mark as unlabeled for SSL
        gt_rest["label"] = -1
        examples_onehot = pd.concat([extended_examples_onehot, extended_examples_onehot_neg, gt_rest]).sample(frac=1)        

        # benchmark decision tree
        dt_result = run_decision_tree_ssl(examples = examples_onehot, data = dataset_onehot)
        dt_performance = calculate_f1_score(dt_result[0], ground_truth_onehot, dataset_onehot)
        benchmark = pd.concat([benchmark, pd.DataFrame([["dt", query["database"], query["id"], query["difficulty"], dt_performance[0], dt_performance[1], dt_performance[2], dt_result[1]]], columns=["algorithm", "dataset", "query", "difficulty", "f1_score", "precision", "recall", "time"])], ignore_index=True)

        # benchmark random forest
        rf_result = run_random_forest_ssl(examples = examples_onehot, data = dataset_onehot)
        rf_performance = calculate_f1_score(rf_result[0], ground_truth_onehot, dataset_onehot)
        benchmark = pd.concat([benchmark, pd.DataFrame([["rf", query["database"], query["id"], query["difficulty"], rf_performance[0], rf_performance[1], rf_performance[2], rf_result[1]]], columns=["algorithm", "dataset", "query", "difficulty", "f1_score", "precision", "recall", "time"])], ignore_index=True)

        # benchmark gradient boosting
        gb_result = run_gradient_boosting_ssl(examples = examples_onehot, data = dataset_onehot)
        gb_performance = calculate_f1_score(gb_result[0], ground_truth_onehot, dataset_onehot)
        benchmark = pd.concat([benchmark, pd.DataFrame([["gb", query["database"], query["id"], query["difficulty"], gb_performance[0], gb_performance[1], gb_performance[2], gb_result[1]]], columns=["algorithm", "dataset", "query", "difficulty", "f1_score", "precision", "recall", "time"])], ignore_index=True)

    return benchmark
