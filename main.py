import argparse
from pathlib import Path
from statistics import mean

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
CV_SCORING = "neg_root_mean_squared_error"


def print_cv_summary(name, train_scores, validation_scores):
    # scikit-learn puts a minus sign on loss scores so larger scores are better.
    folds = ", ".join(f"{-score:.2f}" for score in validation_scores)
    print(f"{name:22} {-mean(train_scores):10.2f} {-mean(validation_scores):12.2f}  {folds}")


def save_prediction_plot(y_test, predictions):
    from matplotlib.figure import Figure

    lower = min(y_test.min(), *(values.min() for values in predictions.values())) - 10
    upper = max(y_test.max(), *(values.max() for values in predictions.values())) + 10
    limits = (lower, upper)

    figure = Figure(figsize=(8, 8), layout="constrained")
    axes = figure.subplots(2, 2, sharex=True, sharey=True)
    for axis, (name, values) in zip(axes.flat, predictions.items(), strict=True):
        axis.plot(limits, limits, color="#424b57", linestyle="--", linewidth=1, zorder=1)
        axis.scatter(y_test, values, s=21, color="#006b96", alpha=0.75, zorder=2)
        axis.set(title=name, xlim=limits, ylim=limits, aspect="equal")

    figure.suptitle(f"Diabetes progression on the test set (n = {len(y_test)})")
    figure.supxlabel("Observed progression score")
    figure.supylabel("Predicted progression score")
    figure.savefig(Path(__file__).with_name("predictions.png"), dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Compare diabetes regression models.")
    parser.add_argument("--plot", action="store_true", help="Save predictions.png using Matplotlib.")
    args = parser.parse_args()

    x, y = load_diabetes(scaled=False, return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    forest = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    models = {
        "Linear regression": LinearRegression(),
        "Random forest": forest,
    }
    candidates = {
        "Ridge": (
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": [0.01, 0.1, 1, 10, 30, 100, 300]},
        ),
        "Tuned random forest": (
            forest,
            {"max_depth": [None, 3, 5, 8, 12], "min_samples_leaf": [1, 3, 5, 10]},
        ),
    }

    print(f"Samples: {len(x_train)} train, {len(x_test)} test. Features: {x.shape[1]}")
    print("\nFive-fold CV on the training split")
    print(f"{'Model':22} {'Train RMSE':>10} {'Val. RMSE':>12}  Validation RMSE by fold")
    for name, model in models.items():
        scores = cross_validate(
            model, x_train, y_train, cv=cv, scoring=CV_SCORING,
            return_train_score=True, error_score="raise",
        )
        print_cv_summary(name, scores["train_score"], scores["test_score"])
        model.fit(x_train, y_train)

    best_params = {}
    for name, (model, param_grid) in candidates.items():
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring=CV_SCORING,
            return_train_score=True, error_score="raise",
        )
        search.fit(x_train, y_train)
        # GridSearchCV refits the best candidate on the full training split.
        models[name] = search.best_estimator_
        best_params[name] = search.best_params_
        results = search.cv_results_
        best = search.best_index_
        train_scores = [results[f"split{fold}_train_score"][best] for fold in range(cv.n_splits)]
        validation_scores = [results[f"split{fold}_test_score"][best] for fold in range(cv.n_splits)]
        print_cv_summary(name, train_scores, validation_scores)

    for name, params in best_params.items():
        print(f"{name} parameters: {params}")

    print("\nHoldout test results")
    print(f"{'Model':22} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    predictions = {}
    for name, model in models.items():
        predicted = model.predict(x_test)
        predictions[name] = predicted
        mae = mean_absolute_error(y_test, predicted)
        rmse = root_mean_squared_error(y_test, predicted)
        r2 = r2_score(y_test, predicted)
        print(f"{name:22} {mae:8.2f} {rmse:8.2f} {r2:8.3f}")

    if args.plot:
        save_prediction_plot(y_test, predictions)
        print("\nSaved predictions.png next to main.py.")


if __name__ == "__main__":
    main()
