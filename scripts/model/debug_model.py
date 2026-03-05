# Testing functions for given phenotype - training only - NO validation

phenotype = 'c_diff'

y, X = load_and_filter_train(phenotype, notes=True, nlp=True, train=True, noisy_label=True)

params = {'C': [0.1, 1], 'max_iter': [200]}
best_model, best_score, features = train_model(X, y, params)
test = fetch_feature_labels(phenotype, features, train=True, noisy_label=True, notes=True, nlp=True)

y_score = best_model.predict_proba(X)[:, 1]
plot_roc_curve(y, y_score, phenotype)


# Evaluating
#TODO: Implement evaluation function

load_and_filter_val
supervised_feature_reduction
train_all


# Evaluate other learning models
# Optional


eval = evaluate_classification_model(best_model, X, y, phenotype)
