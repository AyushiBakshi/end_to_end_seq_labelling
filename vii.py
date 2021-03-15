from main import create_train_model

CNN_CNND = create_train_model("CNN", "CNN_DILATED", 1, "CNND-CNN")
print(CNN_CNND)

vals = [lis[1] for lis in CNN_CNND]
best_val_score = max(vals)
print('Best Val Score:', best_val_score)

best_val_index = vals.index(max(vals))
best_test_score_corr_best_val = CNN_CNND[best_val_index][2]
print('Best Test Score', best_test_score_corr_best_val)

