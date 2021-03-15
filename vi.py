from main import create_train_model

CNN_CNN2 = create_train_model("CNN", "CNN3", 1, 'CNN_CNN_3_L')
print(CNN_CNN2)
vals = [lis[1] for lis in CNN_CNN2]
best_val_score = max(vals)
print('Best Val Score:', best_val_score)

best_val_index = vals.index(max(vals))
best_test_score_corr_best_val = CNN_CNN2[best_val_index][2]
print('Best Test Score', best_test_score_corr_best_val)

# CNN_CNN3 = create_train_model("CNN", "CNN3", 1)
# print(CNN_CNN3)