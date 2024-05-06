import numpy as np
from collections import Counter
from aux_functions import entropy
from sklearn.model_selection import train_test_split, accuracy


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    

class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0): # ADDED FEAT_IDXS HERE
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        
        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
       
        
        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        if np.array_equal(y[left_idxs], np.array([])) == True or np.array_equal(y[right_idxs], np.array([])) == True:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        print(f'Split feature: {best_feat} | Threshold: {best_thresh} | feat_idxs: {feat_idxs}') ####################
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        #print(feat_idxs)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
                        
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
            counter = Counter(y)
            most_common = counter.most_common(1)[0][0]
            return most_common
    

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]



def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    """This random forest have a new variable (patience) where if the model train the patience number of trees without 
     growing the accuracy, the model will stop train decision trees """
    def __init__(self, n_trees=10, min_samples_split=2,
                 max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
    
    #create a validation group
    def fit(self, X, y, patience=None):
        self.trees = []
        
        if patience != None:
            ## Adapted random forest
            #########################
            #########################
            X_train2, X_val, y_train2, y_val = train_test_split(X, y, test_size = 0.15) #create validation set
            
            trees_val_pred=np.empty((X_val.shape[0],0)) #create empy array to add trees preds
            max_val_acc = 0
            
            count_patience = 0
            count_tree = 0
            for _ in range(self.n_trees):
                count_tree +=1 #count trees
                count_patience += 1 #count patience
                if count_patience > patience: #test if patience has been exceeded
                    print (f'stop at {count_tree} trees')
                    return count_tree
                    break
                    
                #create tree
                tree = DecisionTree(min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth, n_feats=self.n_feats)
                X_samp, y_samp = bootstrap_sample(X_train2, y_train2)
                tree.fit(X_samp, y_samp)
                

                #test random forest accuracy with new tree
                trees_val_pred = np.concatenate((trees_val_pred, tree.predict(X_val).reshape((X_val.shape[0], -1))), axis=1) #predict result for tree and add to "list"
                y_val_pred = [most_common_label(tree_pred) for tree_pred in trees_val_pred] #pred the model with actual trees
                val_acc = accuracy(y_val, y_val_pred) #evaluate the model
                
                if val_acc > max_val_acc: #Has the model improved?
                    max_val_acc = val_acc 
                    count_patience = 0
                    
                self.trees.append(tree)
            
        else:
            ##normal random forest
            for _ in range(self.n_trees):
                tree = DecisionTree(min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth, n_feats=self.n_feats)
                # #accuracy
                # count +=1
                # print (count)
                X_samp, y_samp = bootstrap_sample(X, y)
                tree.fit(X_samp, y_samp)
                self.trees.append(tree)


    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)