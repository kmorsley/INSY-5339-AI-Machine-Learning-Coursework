# Homework 7 — GitHub Copilot Code Review

**Student:** Keishana Morsley  
**UTA ID:** [ENTER YOUR UTA ID]  
**Assignment:** Homework 7 - Classification and Regression Trees

> **Before submitting:** This is an editable review worksheet, **not a factual claim that Copilot suggested the passages below**. The assignment requires *three real examples from your own GitHub Copilot session*. Run/open the Python script in VS Code, ask Copilot to review it, and replace the bracketed prompts and the example descriptions with what Copilot actually suggested and what you personally changed.

## Change 1 — Classification tree complexity and overfitting

- **What Copilot suggested:** [Paste or accurately summarize the original Copilot suggestion.]
- **What I changed:** [Describe your actual revision. For example, limiting candidate tree depths to 3, 5, 7 and fully grown, comparing train/test accuracy and nodes/leaves, or preferring a simpler tree when test accuracy is similar.]
- **Why the final version is better:** Evaluating training and testing accuracy alongside tree size makes overfitting visible. The unrestricted classification model had training accuracy 1.0000 but testing accuracy 0.7154; the depth-5 model achieved testing accuracy 0.7308 with fewer nodes.

## Change 2 — Gini versus entropy as a controlled comparison

- **What Copilot suggested:** [Paste or accurately summarize the original Copilot suggestion.]
- **What I changed:** [Describe your actual revision. For example, fixing `max_depth=5` and `random_state=42` for both models, with only the `criterion` changing.]
- **Why the final version is better:** Holding the other settings constant makes the comparison about impurity criterion rather than a confounded change in complexity. On this test split, both criteria achieved accuracy 0.7308, although precision, recall and F1 differed.

## Change 3 — Regression evaluation with held-out data

- **What Copilot suggested:** [Paste or accurately summarize the original Copilot suggestion.]
- **What I changed:** [Describe your actual revision. For example, adding separate non-stratified regression splitting, testing R² and RMSE, depth-based comparison, or residual calculations.]
- **Why the final version is better:** Regression targets are continuous, so classification accuracy and stratification are inappropriate. Testing R² and RMSE reveal how well a model generalizes. The depth-7 model yielded test R² 0.7933 and RMSE 2.8953 (thousand dollars), outperforming the fully grown tree on held-out R².

## Final verification

- [ ] I actually used GitHub Copilot in VS Code.
- [ ] Each **What Copilot suggested** section accurately reflects a suggestion I received.
- [ ] Each **What I changed** section describes a change I actually made.
- [ ] I reviewed the Python script, executed it successfully and confirmed generated outputs.
- [ ] I replaced `UTAID` in filenames with my university identification number.
