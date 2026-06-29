import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


def build_dnn_module(input_dim):
    """
    Build a Deep Neural Network
    """
    #we use sequential model
    model = Sequential([
        Dense(256, input_dim=input_dim),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),

        #layer 2
        Dense(128),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),

        #layer 3
        Dense(64),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),

        #layer 4
        Dense(32),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),

        #layer 5
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_all_models():
    """
    Returns a dict to every ML model
    """
    return {
        'XGBoost': xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_ratie=0.05, eval_matrix='logloss', random_state=42),

        'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', max_features='sqrt', random_state=42),

        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),

        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),

        'SVM': SVC(probability=True, random_state=42),

        'KNN': KNeighborsClassifier(n_neighbors=5),

        'Gaussian Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    }

def build_stacking_ensemble(models_dict):
    """
    Builds a strong voting classifier
    """

    estimators = [
        ('rf', models_dict['Random Forest']),
        ('xb', models_dict['XGBoost']),
        ('lr', models_dict['LogisticRegression'])
    ]
    return VotingClassifier(estimators=estimators, voting='soft')

def master_training_orchestrator(x,y):
    """
    The master training loop
    """
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

    metrics={}
    trained_models={}

    print("Training 1/12: Deep Neural Network")
    dnn_model = build_dnn_module(input_dim=X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    dnn_model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

    dnn_preds_prob = dnn_model.predict(X_val).flatten()
    dnn_preds = (dnn_preds_prob > 0.5).astype(int)

    metrics['DNN'] = {
        'Accuracy': float(accuracy_score(y_val, dnn_preds)),
        'F1-Score': float(f1_score(y_val, dnn_preds)),
        'Precision': float(precision_score(y_val, dnn_preds)),
        'Recall': float(recall_score(y_val, dnn_preds)),
        'ROC-AUC': float(roc_auc_score(y_val, dnn_preds_prob)),
        'Confusion_Matrix': confusion_matrix(y_val, dnn_preds).tolist()
    }
    trained_models['DNN'] = dnn_model

    #training models
    ml_models = get_all_models()

    #go through all
    for i, (name,model) in enumerate(ml_models.items(), start=2):
        print(f"Training {i}/12: {name}")

        #train the models
        model.fit(X_train, y_train)

        #make pred
        preds = model.predict(X_val)

        #calc probs
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_val)[:,1]
            roc_auc = float(roc_auc_score(y_val, probs))
        else:
            roc_auc = float(roc_auc_score(y_val, preds))

        metrics[name] = {
            'Accuracy': float(accuracy_score(y_val, preds)),
            'F1-Score': float(f1_score(y_val, preds)),
            'Precision': float(precision_score(y_val, preds, zero_division=0)),
            'Recall': float(recall_score(y_val, preds, zero_division=0)),
            'ROC-AUC': roc_auc,
            'Confusion_Matrix': confusion_matrix(y_val, preds).tolist()
        }
        trained_models[name] = model

        #training vote
        print("Training 12/12: Soft voting ensemble")
        ensemble = build_stacking_ensemble(ml_models)
        ensemble.fit(X_train,y_train)

        ens_preds = ensemble.predict(X_val)
        ens_probs = ensemble.predict_proba(X_val)[:,1]

        metrics['Voting Ensemble'] = {
            'Accuracy': float(accuracy_score(y_val, ens_preds)),
            'F1-Score': float(f1_score(y_val, ens_preds)),
            'Precision': float(precision_score(y_val, ens_preds)),
            'Recall': float(recall_score(y_val, ens_preds)),
            'ROC-AUC': float(roc_auc_score(y_val, ens_probs)),
            'Confusion_Matrix': confusion_matrix(y_val, ens_preds).tolist()
        }
        trained_models['Voting Ensemble'] = ensemble

        print("All 12 trained and evaluated successfully")


        #we return architecture
        return trained_models, metrics, X_train, X_val
