from algorithms.squid import prepare_squid, run_squid
from algorithms.decision_tree import run_decision_tree
from algorithms.random_forest import run_random_forest
from algorithms.gradient_boosting import run_gradient_boosting
import pandas as pd
from helper_functions import calculate_f1_score, translate_query_for_onehot_table, build_base_query


# Run second experiment: Supervised Learning, Balanced Dataset, with Base Query extension for DT, RF, GB, SQuID
def run_experiment_2(query_set, datasets, datasets_onehot):

    print("----------------------------------------")
    print("Experiment 2: Supervised Learning, Balanced Dataset, with Base Query extension")
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
        # list columns in the query. In a normal QBE setting these would be known by the user provided example columns.
        base_query_column_subset = [word for word in examples.columns.values if word in str(query["query"]).split(" ")]
        # construct base query on this column subset
        base_query = build_base_query(example=examples[base_query_column_subset])
        # get extended example set using the constructed base query
        extended_examples = dataset.query(base_query)
        query_for_onehot = translate_query_for_onehot_table(query["query"])
        base_query_for_onehot = translate_query_for_onehot_table(base_query)
        print(query_for_onehot)
        print(base_query)
        print(base_query_for_onehot)
        # construct negatvie base query as negated postivie base query
        base_query_for_onehot_neg = "not (" + base_query_for_onehot + ")"
        ground_truth_onehot = dataset_onehot.query(query_for_onehot)
        extended_examples_onehot = dataset_onehot.query(base_query_for_onehot).copy()        
        extended_examples_onehot_neg = dataset_onehot.query(base_query_for_onehot_neg).copy()

        # balance the pos. and neg. example set size by oversampling the smaller set
        if(len(extended_examples_onehot) < len(extended_examples_onehot_neg)):
            extended_examples_onehot = extended_examples_onehot.sample(
                len(extended_examples_onehot_neg), replace=True)
        else:
            extended_examples_onehot_neg = extended_examples_onehot_neg.sample(
                len(extended_examples_onehot), replace=True)

        # set the labels
        extended_examples_onehot["label"] = 1
        extended_examples_onehot_neg["label"] = 0
        examples_onehot = pd.concat([extended_examples_onehot, extended_examples_onehot_neg]).sample(frac=1)        

        # benchmark SQuID
        squid_result = run_squid(examples = extended_examples, db_name = str(query["database"]))
        squid_perfomance = calculate_f1_score(squid_result[0], ground_truth, dataset, squid=True)
        benchmark = pd.concat([benchmark, pd.DataFrame([["squid", query["database"], query["id"], query["difficulty"], squid_perfomance[0], squid_perfomance[1], squid_perfomance[2], squid_result[1]]], columns=["algorithm", "dataset", "query", "difficulty", "f1_score", "precision", "recall", "time"])], ignore_index=True)

        # benchmark decision tree
        dt_result = run_decision_tree(examples = examples_onehot, data = dataset_onehot)
        dt_performance = calculate_f1_score(dt_result[0], ground_truth_onehot, dataset_onehot)
        benchmark = pd.concat([benchmark, pd.DataFrame([["dt", query["database"], query["id"], query["difficulty"], dt_performance[0], dt_performance[1], dt_performance[2], dt_result[1]]], columns=["algorithm", "dataset", "query", "difficulty", "f1_score", "precision", "recall", "time"])], ignore_index=True)

        # benchmark random forest
        rf_result = run_random_forest(examples = examples_onehot, data = dataset_onehot)
        rf_performance = calculate_f1_score(rf_result[0], ground_truth_onehot, dataset_onehot)
        benchmark = pd.concat([benchmark, pd.DataFrame([["rf", query["database"], query["id"], query["difficulty"], rf_performance[0], rf_performance[1], rf_performance[2], rf_result[1]]], columns=["algorithm", "dataset", "query", "difficulty", "f1_score", "precision", "recall", "time"])], ignore_index=True)

        # benchmark gradient boosting
        gb_result = run_gradient_boosting(examples = examples_onehot, data = dataset_onehot)
        gb_performance = calculate_f1_score(gb_result[0], ground_truth_onehot, dataset_onehot)
        benchmark = pd.concat([benchmark, pd.DataFrame([["gb", query["database"], query["id"], query["difficulty"], gb_performance[0], gb_performance[1], gb_performance[2], gb_result[1]]], columns=["algorithm", "dataset", "query", "difficulty", "f1_score", "precision", "recall", "time"])], ignore_index=True)

    return benchmark
