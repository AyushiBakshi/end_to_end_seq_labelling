from main import create_train_model

CNN_CNN = create_train_model("CNN", "CNN", 1, 'CNN_CNN_1_L')
print(CNN_CNN)

vals = [lis[1] for lis in CNN_CNN]
best_val_score = max(vals)
print('Best Val Score:', best_val_score)

best_val_index = vals.index(max(vals))
best_test_score_corr_best_val = CNN_CNN[best_val_index][2]
print('Best Test Score', best_test_score_corr_best_val)
