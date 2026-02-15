import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Load dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split dataset
    print("Splitting data into 80% train and 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    # Train and evaluate
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
        report = classification_report(y_test, y_pred, target_names=iris.target_names)
        
        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)
        print("-" * 60)
        
        # Collect results for CSV
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (weighted)": report_dict['weighted avg']['precision'],
            "Recall (weighted)": report_dict['weighted avg']['recall'],
            "F1-Score (weighted)": report_dict['weighted avg']['f1-score']
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
